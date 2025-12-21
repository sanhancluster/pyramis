import os
import h5py
import numpy as np

from concurrent.futures import as_completed
import warnings

from . import get_config, get_dim_keys, get_vname
from .core import compute_chunk_list_from_hilbert
from .geometry import Region, Box
from .utils.arrayview import SharedView
from .utils import get_mp_executor
from. import io

from multiprocessing.shared_memory import SharedMemory

config = get_config()

def get_by_type(obj: h5py.File | h5py.Group, name:str, datatype=None):
    data = obj.get(name)
    if datatype is not None:
        assert isinstance(data, datatype), f"{name} is not of type {datatype}"
    return data


def remap_dtype_fields(dtype: np.dtype, field_mapping: dict | None=None) -> np.dtype:
    """
    Create a new dtype by renaming fields of an existing compound dtype.
    Parameters
    ----------
    dtype : np.dtype
        Original compound dtype.
    field_mapping : dict
        Mapping from old field names to new field names.
    Returns
    -------
    np.dtype
        New compound dtype with renamed fields.
    """
    if field_mapping is None:
        field_mapping = config['VNAME_MAPPING'][config['VNAME_GROUP']]
    new_fields = []
    for name in dtype.names:
        if name in field_mapping and field_mapping[name] is not None:
            new_name = field_mapping[name]
        else:
            new_name = name
        new_fields.append((new_name, dtype.fields[name][0]))
    new_dtype = np.dtype(new_fields)
    return new_dtype


def _chunk_size_worker(
        args):
    path, name, start, end, region, is_cell = args

    with h5py.File(path, 'r', locking=False) as f:
        group = get_by_type(f, name, h5py.Group)
        
        if region is None:
            return group.attrs['size']
            
        data = get_by_type(group, 'data', h5py.Dataset)
        dtype = data.dtype

        fields_native = get_dim_keys(name_group='native')
        if is_cell:
            fields_native = fields_native + ['level']

        new_dtype = np.dtype([(name, dtype.fields[name][0]) for name in fields_native if name in dtype.names])
        new_dtype = remap_dtype_fields(new_dtype)
    
        data_slice = data.fields(fields_native)[start:end].view(new_dtype)
        boxsize = f.attrs.get('boxsize', 1.0)
        mask = region.contains_data(data_slice, cell=is_cell, boxsize=boxsize)

        return np.sum(mask)


def _load_slice_worker(args):
    """
    Worker that reads a slice from an HDF5 dataset and writes it directly
    into a shared memory NumPy array.

    Parameters
    ----------
    args : tuple
        (
            path,           # HDF5 file path
            group_name,     # HDF5 group containing 'data'
            target_fields,  # None or list of field names for compound dtype
            shm_name,       # name of existing SharedMemory block
            total_len,      # total number of rows in the final array
            shape_tail,     # trailing shape (dataset.shape[1:])
            dtype_str,      # dtype as string (e.g. '<f8')
            start,          # slice start index
            end,            # slice end index
            offset          # where to write in the shared array
        )
    """

    (path, group_name, target_fields_native,
     shm_name, shared_arr, ndata_tot, dtype_out,
     start, end, offset, ndata, region, is_cell) = args

    # Each worker opens the HDF5 file independently.
    with h5py.File(path, 'r', locking=False) as f:
        group = f.get(group_name)
        data = group.get('data')
        if target_fields_native is not None:
            # If dataset is compound, select only requested fields.
            data = data.fields(target_fields_native)

        data_slice = data[start:end].view(dtype_out)
        # Precompute mask if needed
        if region is not None:
            boxsize = f.attrs.get('boxsize', 1.0)
            mask = region.contains_data(data_slice, cell=is_cell, boxsize=boxsize)
        else:
            mask = None

        if shm_name is not None:
            shm = SharedMemory(name=shm_name)
            try:
                target = np.ndarray((ndata_tot,), dtype=dtype_out, buffer=shm.buf)
                if mask is not None:
                    target[offset:offset + ndata] = data_slice[mask]
                else:
                    target[offset:offset + ndata] = data_slice
            finally:
                # Worker should only close its handle, never unlink the shared memory.
                shm.close()
        else:
            # shared_arr is provided by the parent when not using SharedMemory
            if mask is not None:
                shared_arr[offset:offset + ndata] = data_slice[mask]
            else:
                print(data_slice.dtype, shared_arr.dtype)
                shared_arr[offset:offset + ndata] = data_slice

def _chunk_slice_hdf_mp(
    path,
    group_name,
    chunk_indices,
    chunk_sizes=1,
    region: Region | None=None,
    boundary_name="chunk_boundary",
    target_fields=None,
    n_workers=config['DEFAULT_N_PROCS'],
    mp_backend="process",
    copy_result=True,
    is_cell=False
):

    chunk_indices = np.asarray(chunk_indices)
    if np.isscalar(chunk_sizes):
        chunk_sizes = np.full_like(chunk_indices, int(chunk_sizes))
    else:
        chunk_sizes = np.asarray(chunk_sizes)

    # Read only meta-info once in the parent
    with h5py.File(path, "r") as f:
        group = f[group_name]
        data = get_by_type(group, "data", h5py.Dataset)
        dtype = remap_dtype_fields(data.dtype)
        if target_fields is not None:
            mapping = config['VNAME_MAPPING'][config['VNAME_GROUP']]
            mapping_reverse = {v: k for k, v in mapping.items() if isinstance(v, str)}
            target_fields_native = [mapping_reverse.get(f, f) for f in target_fields]
            data = data.fields(target_fields_native)
        else:
            target_fields_native = None

        bounds = get_by_type(group, boundary_name, h5py.Dataset)[:]
    
    if target_fields is not None:
        dtype_out = np.dtype([(name, dtype.fields[name][0]) for name in target_fields if name in dtype.names])
    else:
        dtype_out = dtype

    # Compute [start, end) for each chunk
    starts = bounds[chunk_indices]
    ends = bounds[chunk_indices + chunk_sizes]

    if region is not None:
        # Compute exact sizes by filtering with region in parallel
        jobs = [
            (path, group_name, int(start), int(end), region, is_cell)
            for start, end in zip(starts, ends)
        ]
        with get_mp_executor(backend=mp_backend, n_workers=n_workers) as executor:
            futures = [executor.submit(_chunk_size_worker, args) for args in jobs]

            # Gather sizes
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc is not None:
                    # Raise the first error encountered
                    raise exc
        # Reorder sizes to match chunk order
        ndata_per_chunk = np.array([fut.result() for fut in futures])
    else:
        ndata_per_chunk = ends - starts
    ndata_tot = int(np.sum(ndata_per_chunk))

    if ndata_tot == 0:
        return np.empty((0,), dtype=dtype_out)

    # Pre-compute offsets so each worker writes to a unique region
    offsets = np.zeros_like(ndata_per_chunk)
    offsets[1:] = np.cumsum(ndata_per_chunk[:-1])
    offsets = offsets.astype(int)

    # Allocate shared memory for the entire final array
    itemsize = dtype.itemsize
    total_bytes = ndata_tot * itemsize

    if mp_backend == "process" and n_workers > 1:
        shm = SharedMemory(create=True, size=total_bytes)
        try:
            shared_arr = np.ndarray((ndata_tot, ), dtype=dtype_out, buffer=shm.buf)

            # Prepare worker job arguments
            jobs = [
                (path, group_name, target_fields_native, shm.name, None, ndata_tot, dtype_out, int(start), int(end), int(offset), int(ndata), region, is_cell)
                for start, end, offset, ndata in zip(starts, ends, offsets, ndata_per_chunk)
                if ndata > 0]

            with get_mp_executor(backend=mp_backend, n_workers=n_workers) as executor:
                futures = [executor.submit(_load_slice_worker, args) for args in jobs]

                # Propagate the first exception (if any)
                for fut in as_completed(futures):
                    exc = fut.exception()
                    if exc is not None:
                        # Raise the first error encountered
                        raise exc

            if copy_result:
                result = np.array(shared_arr, copy=True)
            else:
                result = SharedView(shm, (ndata_tot,), dtype)
                
        finally:
            if copy_result:
                try:
                    shm.close()
                except FileNotFoundError:
                    pass
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
    else:
        shared_arr = np.empty((ndata_tot,), dtype=dtype_out)
        jobs = [
            (path, group_name, target_fields_native, None, shared_arr, ndata_tot, dtype_out, int(start), int(end), int(offset), int(size), region, is_cell)
            for start, end, offset, size in zip(starts, ends, offsets, ndata_per_chunk)
            if size > 0]
        with get_mp_executor(backend=mp_backend, n_workers=n_workers) as executor:
            futures = [executor.submit(_load_slice_worker, args) for args in jobs]

            # Propagate the first exception (if any)
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc is not None:
                    # Raise the first error encountered
                    raise exc
        result = shared_arr        

    return result

def _chunk_slice_hdf(
        path, 
        name:str, 
        chunk_indices, 
        chunk_sizes=1,
        region: Region | None=None,
        boundary_name='chunk_boundary', 
        target_fields=None,
        is_cell=False) -> np.ndarray:

    with h5py.File(path, 'r') as f:
        group = get_by_type(f, name, h5py.Group)
        data = get_by_type(group, 'data', h5py.Dataset)
        dtype = remap_dtype_fields(data.dtype)
        bounds = get_by_type(group, boundary_name, h5py.Dataset)[:]
        starts, ends = bounds[chunk_indices], bounds[chunk_indices + chunk_sizes]

        if target_fields is not None:
            mapping = config['VNAME_MAPPING'][config['VNAME_GROUP']]
            mapping_reverse = {v: k for k, v in mapping.items() if isinstance(v, str)}
            target_fields_native = [mapping_reverse.get(f, f) for f in target_fields]
            data = data.fields(target_fields_native)

        if target_fields is not None:
            dtype_out = np.dtype([(name, dtype.fields[name][0]) for name in target_fields if name in dtype.names])
        else:
            dtype_out = dtype

        # Read and filter each chunk
        boxsize = f.attrs.get('boxsize', 1.0)
        output_list = []
        for start, end in zip(starts, ends):
            data_slice = data[start:end].view(dtype_out)
            if region is not None:
                mask = region.contains_data(data_slice, cell=is_cell, boxsize=boxsize)
                data_slice = data_slice[mask]
            output_list.append(data_slice)
        output = np.concatenate(output_list)
    return output


def read_hdf(
        filename, 
        name:str, 
        region: Region | np.ndarray | list | None=None, 
        target_fields=None, 
        levelmax=None, 
        levelmin=None,
        exact_cut=True,
        n_workers=config['DEFAULT_N_PROCS'],
        use_process=True,
        copy_result=True,
        is_cell=False):

    if use_process:
        mp_backend = "process"
    else:
        mp_backend = "thread"
    
    if exact_cut and region is not None and target_fields is not None:
        warn = False
        dim_keys = get_dim_keys()
        for key in dim_keys:
            if key not in target_fields:
                warn = True
                target_fields = target_fields + [key]

        vname_level = get_vname('level')
        if vname_level not in target_fields and is_cell:
            warn = True
            target_fields = target_fields + [vname_level]

        if warn:
            warnings.warn("Exact cut with region specified requires position fields to be loaded. They have been added to target_fields.")

    if isinstance(region, np.ndarray) or isinstance(region, list):
        region = Box(region)

    with h5py.File(filename, 'r') as f:
        group = get_by_type(f, name, h5py.Group)
        if region is not None:
            chunk_indices = compute_chunk_list_from_hilbert(
                region=region,
                hilbert_boundary=get_by_type(group, 'hilbert_boundary', h5py.Dataset)[:],
                level_hilbert=group.attrs.get('levelmax', 1),
                boxlen=group.parent.attrs.get('boxlen', 1.0),
            )
        else:
            nchunks = int(group.attrs.get('n_chunk', 0))
            chunk_indices = np.arange(nchunks)
        if levelmax is not None or levelmin is not None:
            if levelmin is None:
                levelmin = 1
            if levelmax is None:
                levelmax = group.attrs.get('levelmax')
            level_indices = chunk_indices * group.attrs.get('n_level', 1)
            chunk_indices = level_indices + (levelmin - 1)
            chunk_sizes = levelmax - levelmin + 1
        else:
            chunk_sizes = 1

    if n_workers == 1:
        if not exact_cut:
            region = None
        result = _chunk_slice_hdf(filename, name, chunk_indices, chunk_sizes=chunk_sizes, region=region, target_fields=target_fields, is_cell=is_cell)
    else:
        if not exact_cut:
            region = None
        result = _chunk_slice_hdf_mp(filename, name, chunk_indices, chunk_sizes=chunk_sizes, region=region, target_fields=target_fields, n_workers=n_workers, mp_backend=mp_backend, copy_result=copy_result, is_cell=is_cell)
    return result


def read_part(
        path: str,
        part_type: str,
        iout: int | None=None,
        region: Region | np.ndarray | list | None=None,
        target_fields=None,
        exact_cut=True,
        n_workers=config['DEFAULT_N_PROCS'],
        use_process=True,
        copy_result=True):

    if iout is None:
        filename = path
    else:
        filename = os.path.join(path, config['FILENAME_FORMAT_HDF'].format(data='part', iout=iout))
    data = read_hdf(filename, part_type, region=region, target_fields=target_fields, exact_cut=exact_cut, n_workers=n_workers, use_process=use_process, copy_result=copy_result, is_cell=False)
    return data


def read_cell(
        path: str,
        iout: int | None=None,
        region: Region | np.ndarray | list | None=None,
        target_fields=None,
        levelmax_load=None,
        exact_cut=True,
        n_workers=config['DEFAULT_N_PROCS'],
        use_process=True,
        copy_result=True,
        read_branch=False):
    
    if iout is None:
        filename = path
    else:
        filename = os.path.join(path, config['FILENAME_FORMAT_HDF'].format(data='cell', iout=iout))
    if levelmax_load is not None:
        data_leaf = read_hdf(filename, 'branch', region=region, target_fields=target_fields, exact_cut=exact_cut, n_workers=n_workers, levelmax=levelmax_load, use_process=use_process, copy_result=copy_result, is_cell=True)
        data_branch = read_hdf(filename, 'leaf', region=region, target_fields=target_fields, exact_cut=exact_cut, n_workers=n_workers, levelmin=levelmax_load, levelmax=levelmax_load, use_process=use_process, copy_result=copy_result, is_cell=True)
        data = np.concatenate([data_leaf, data_branch])
    elif read_branch:
        data = read_hdf(filename, 'branch', region=region, target_fields=target_fields, exact_cut=exact_cut, n_workers=n_workers, use_process=use_process, copy_result=copy_result, is_cell=True)
    else:
        data = read_hdf(filename, 'leaf', region=region, target_fields=target_fields, exact_cut=exact_cut, n_workers=n_workers, use_process=use_process, copy_result=copy_result, is_cell=True)

    return data


def ramsese_to_hdf(
        path_ramses: str,
        iout: int,
        path_hdf: str,
        n_workers=config['DEFAULT_N_PROCS'],
        use_process=True):

    if use_process:
        mp_backend = "process"
    else:
        mp_backend = "thread"

    part = io.read_part(path_ramses, iout, n_workers=n_workers, use_process=use_process, copy_result=False)
    cell = io.read_cell(path_ramses, iout, n_workers=n_workers, use_process=use_process, copy_result=False)

    with h5py.File(os.path.join(path_hdf, config['FILENAME_FORMAT_HDF'].format(data='part', iout=iout)), 'w') as f:
        group = f.create_group('part')
        dset = group.create_dataset('data', data=part, compression="gzip")
        group.attrs['nchunks'] = 1
        group.attrs['hilbert_boundary'] = np.array([[0]], dtype=np.uint64)

    with h5py.File(os.path.join(path_hdf, config['FILENAME_FORMAT_HDF'].format(data='cell', iout=iout)), 'w') as f:
        group_leaf = f.create_group('leaf')
        dset_leaf = group_leaf.create_dataset('data', data=cell['leaf'], compression="gzip")
        group_leaf.attrs['nchunks'] = 1
        group_leaf.attrs['hilbert_boundary'] = np.array([[0]], dtype=np.uint64)

        group_branch = f.create_group('branch')
        dset_branch = group_branch.create_dataset('data', data=cell['branch'], compression="gzip")
        group_branch.attrs['nchunks'] = 1
        group_branch.attrs['hilbert_boundary'] = np.array([[0]], dtype=np.uint64)