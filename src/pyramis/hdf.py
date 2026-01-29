import os
import h5py
import numpy as np
import glob

from concurrent.futures import as_completed
import warnings

from . import config, get_dim_keys, get_vname, get_mapping, cgs_unit
from .core import compute_chunk_list_from_hilbert
from .geometry import Region, Box
from .utils.arrayview import SharedView
from .utils import get_mp_executor
from. import io
from .astro import get_cosmo_table, cosmo_convert

from multiprocessing.shared_memory import SharedMemory


def check_snapshots(path: str, check_data=['cell', 'part']) -> np.ndarray:
    iout_list = None
    for data in check_data:
        pattern = config['FILENAME_FORMAT_HDF_ANY'].format(data=data)
        files = glob.glob(os.path.join(path, pattern))
        iouts_data = []
        for f in files:
            basename = os.path.basename(f)
            parts = basename.split('_')
            if len(parts) < 2:
                continue
            iout_part = parts[1]
            try:
                iout_str = iout_part.split('.')[0]
                iout = int(iout_str)
                iouts_data.append(iout)
            except ValueError:
                continue
        if iout_list is None:
            iout_list = np.array(iouts_data)
        else:
            iout_list = iout_list[np.isin(iout_list, iouts_data)]
    if iout_list is None:
        iout_list = np.array([])
    
    aexp_list, time_list, nstep_coarse_list = [], [], []
    for iout in iout_list:
        info = get_info(path, iout, cosmo=False)
        aexp_list.append(info.get('aexp', 1.0))
        time_list.append(info.get('age', 0.0))
        nstep_coarse_list.append(info.get('icoarse', 0))

    table = np.rec.fromarrays(
        [iout_list, aexp_list, time_list, nstep_coarse_list, np.zeros(len(iout_list), dtype=bool)],
        dtype=[('iout', 'i4'), ('aexp', 'f8'), ('time', 'f8'), ('nstep_coarse', 'i4'), ('scheduled', '?')])
    table = np.sort(table, order='iout')
    return table


def get_by_type(obj: h5py.File | h5py.Group, name:str, datatype=None):
    data = obj.get(name)
    if datatype is not None:
        assert isinstance(data, datatype), f"{name} is not of type {datatype}"
    return data


def remap_dtype_names(dtype: np.dtype, mapping: dict | None=None) -> np.dtype:
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
    if mapping is None:
        mapping = config['VNAME_MAPPING'][config['VNAME_SET']]
    new_fields = []
    for name in dtype.names:
        if name in mapping and mapping[name] is not None:
            new_name = mapping[name]
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

        vname_set_file = f.attrs.get('vname_set', 'native')
        fields_file = get_dim_keys(name_set=vname_set_file)
        if is_cell:
            fields_file = fields_file + ['level']

        new_dtype = np.dtype([(name, dtype.fields[name][0]) for name in fields_file if name in dtype.names])
        new_dtype = remap_dtype_names(new_dtype)
    
        data_slice = data.fields(fields_file)[start:end].view(new_dtype)
        boxsize = f.attrs.get('boxsize', 1.0)
        mask = region.contains_data(data_slice, cell=is_cell, boxlen=boxsize)

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
            mask = region.contains_data(data_slice, cell=is_cell, boxlen=boxsize)
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
    is_cell=False,
    use_vname_mapping=True,
):

    chunk_indices = np.asarray(chunk_indices)
    if np.isscalar(chunk_sizes):
        chunk_sizes = np.full_like(chunk_indices, int(chunk_sizes))
    else:
        chunk_sizes = np.asarray(chunk_sizes)

    # Read only meta-info once in the parent
    with h5py.File(path, "r") as f:
        _, dtype_out, target_fields, starts, ends = _prepare_hdf_read(f, group_name, boundary_name, chunk_indices, chunk_sizes, target_fields, use_vname_mapping)

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
    itemsize = dtype_out.itemsize
    total_bytes = ndata_tot * itemsize

    if mp_backend == "process" and n_workers > 1:
        shm = SharedMemory(create=True, size=total_bytes)
        try:
            shared_arr = np.ndarray((ndata_tot, ), dtype=dtype_out, buffer=shm.buf)

            # Prepare worker job arguments
            jobs = [
                (path, group_name, target_fields, shm.name, None, ndata_tot, dtype_out, int(start), int(end), int(offset), int(ndata), region, is_cell)
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
            (path, group_name, target_fields, None, shared_arr, ndata_tot, dtype_out, int(start), int(end), int(offset), int(size), region, is_cell)
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
        group_name:str, 
        chunk_indices, 
        chunk_sizes=1,
        region: Region | None=None,
        boundary_name='chunk_boundary', 
        target_fields=None,
        is_cell=False,
        use_vname_mapping=True) -> np.ndarray:

    with h5py.File(path, 'r') as f:
        data, dtype_out, target_fields, starts, ends = _prepare_hdf_read(f, group_name, boundary_name, chunk_indices, chunk_sizes, target_fields, use_vname_mapping)

        # Read and filter each chunk
        boxsize = f.attrs.get('boxsize', 1.0)
        output_list = []
        for start, end in zip(starts, ends):
            data_slice = data[start:end].view(dtype_out)
            if region is not None:
                mask = region.contains_data(data_slice, cell=is_cell, boxlen=boxsize)
                data_slice = data_slice[mask]
            output_list.append(data_slice)
        output = np.concatenate(output_list)
    return output


def _prepare_hdf_read(f, group_name, boundary_name, chunk_indices, chunk_sizes, target_fields, use_vname_mapping):
    vname_set_file = f.attrs.get('vname_set', 'native')
    if vname_set_file == config['VNAME_SET']:
        use_vname_mapping = False
    group = get_by_type(f, group_name, h5py.Group)
    data = get_by_type(group, 'data', h5py.Dataset)
    bounds = get_by_type(group, boundary_name, h5py.Dataset)
    starts, ends = bounds[chunk_indices], bounds[chunk_indices + chunk_sizes]

    dtype_file = data.dtype
    mapping = get_mapping(vname_set_file, config['VNAME_SET'])
    if target_fields is not None:
        if use_vname_mapping:
            mapping_reverse = get_mapping(config['VNAME_SET'], vname_set_file)
            target_fields_file = [mapping_reverse.get(f, f) for f in target_fields if mapping_reverse.get(f, f) in dtype_file.names]
        else:
            target_fields_file = target_fields
        data = data.fields(target_fields_file)
        dtype_out = np.dtype([(name, dtype_file.fields[name][0]) for name in target_fields_file if name in dtype_file.names])
    else:
        dtype_out = dtype_file
        target_fields_file = None
    dtype_out = remap_dtype_names(dtype_out, mapping) if use_vname_mapping else dtype_out

    return data, dtype_out, target_fields_file, starts, ends


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
        is_cell=False,
        use_vname_mapping=True):

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
        result = _chunk_slice_hdf(filename, name, chunk_indices, chunk_sizes=chunk_sizes, region=region, target_fields=target_fields, is_cell=is_cell, use_vname_mapping=use_vname_mapping)
    else:
        if not exact_cut:
            region = None
        result = _chunk_slice_hdf_mp(filename, name, chunk_indices, chunk_sizes=chunk_sizes, region=region, target_fields=target_fields, n_workers=n_workers, mp_backend=mp_backend, copy_result=copy_result, is_cell=is_cell, use_vname_mapping=use_vname_mapping)
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
        copy_result=True,
        use_vname_mapping=True):
    """
    Read particle data from HDF5 file.

    Parameters
    ----------
    path : str
        Path to the directory containing HDF5 files or full file path if iout is None.
    part_type : str
        Type of particles to read (e.g., 'dark_matter', 'star', etc.).
    iout : int, optional
        Output number to construct the filename. If None, `path` is treated as the full filename.
    region : Region or array-like, optional
        Region to filter particles. If None, all particles are read.
    target_fields : list, optional
        List of fields to read. If None, all fields are read.
    exact_cut : bool, optional
        Whether to apply exact cut based on the region. Defaults to True. If False, all particles in the chunks overlapping the region are read.
    n_workers : int, optional
        Number of parallel workers to use. Defaults to config['DEFAULT_N_PROCS'].
    use_process : bool, optional
        Whether to use process-based parallelism. Defaults to True. If False, thread-based parallelism is used.
    copy_result : bool, optional
        Whether to return a copy of the result array. Defaults to True. If False, a shared memory view is returned when using multiple processes.
    use_vname_mapping : bool, optional
        Whether to apply variable name mapping. Defaults to True. If False, variable names stored in the file are used directly.
    
    Returns
    -------
    np.ndarray
        Array of particle data.
    """

    if iout is None:
        filename = path
    else:
        filename = os.path.join(path, config['FILENAME_FORMAT_HDF'].format(data='part', iout=iout))
    data = read_hdf(filename, part_type, region=region, target_fields=target_fields, exact_cut=exact_cut, n_workers=n_workers, use_process=use_process, copy_result=copy_result, is_cell=False, use_vname_mapping=use_vname_mapping)
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
        read_branch=False,
        use_vname_mapping=True):
    """
    Read cell data from HDF5 file.

    Parameters
    ----------
    path : str
        Path to the directory containing HDF5 files or full file path if iout is None.
    iout : int, optional
        Output number to construct the filename. If None, `path` is treated as the full filename.
    region : Region or array-like, optional
        Region to filter cells. If None, all cells are read.
    target_fields : list, optional
        List of fields to read. If None, all fields are read.
    levelmax_load : int, optional
        If specified, load cells up to this maximum refinement level from leaf and branch datasets.
    exact_cut : bool, optional
        Whether to apply exact cut based on the region. Defaults to True. If False, all cells in the chunks overlapping the region are read.
    n_workers : int, optional
        Number of parallel workers to use. Defaults to config['DEFAULT_N_PROCS'].
    use_process : bool, optional
        Whether to use process-based parallelism. Defaults to True. If False, thread-based parallelism is used.
    copy_result : bool, optional
        Whether to return a copy of the result array. Defaults to True. If False, a shared memory view is returned when using multiple processes.
    read_branch : bool, optional
        Whether to read branch cells instead of leaf cells. Defaults to False.
    use_vname_mapping : bool, optional
        Whether to apply variable name mapping. Defaults to True. If False, variable names stored in the file are used directly.

    Returns
    -------
    np.ndarray
        Array of cell data.
    """
    
    if iout is None:
        filename = path
    else:
        filename = os.path.join(path, config['FILENAME_FORMAT_HDF'].format(data='cell', iout=iout))
    if levelmax_load is not None:
        data_leaf = read_hdf(filename, 'branch', region=region, target_fields=target_fields, exact_cut=exact_cut, n_workers=n_workers, levelmax=levelmax_load, use_process=use_process, copy_result=copy_result, is_cell=True, use_vname_mapping=use_vname_mapping)
        data_branch = read_hdf(filename, 'leaf', region=region, target_fields=target_fields, exact_cut=exact_cut, n_workers=n_workers, levelmin=levelmax_load, levelmax=levelmax_load, use_process=use_process, copy_result=copy_result, is_cell=True, use_vname_mapping=use_vname_mapping)
        data = np.concatenate([data_leaf, data_branch])
    elif read_branch:
        data = read_hdf(filename, 'branch', region=region, target_fields=target_fields, exact_cut=exact_cut, n_workers=n_workers, use_process=use_process, copy_result=copy_result, is_cell=True, use_vname_mapping=use_vname_mapping)
    else:
        data = read_hdf(filename, 'leaf', region=region, target_fields=target_fields, exact_cut=exact_cut, n_workers=n_workers, use_process=use_process, copy_result=copy_result, is_cell=True, use_vname_mapping=use_vname_mapping)

    return data


def read_star(path: str, iout: int | None=None, region: Region | np.ndarray | list | None=None,
              target_fields=None, exact_cut=True, n_workers=config['DEFAULT_N_PROCS'],
              use_process=True, copy_result=True, use_vname_mapping=True):
    """
    Read star particle data from HDF5 file.
    """
    return read_part(path, 'star', iout=iout, region=region, target_fields=target_fields, 
                     exact_cut=exact_cut, n_workers=n_workers, use_process=use_process, 
                     copy_result=copy_result, use_vname_mapping=use_vname_mapping)

def read_dm(path: str, iout: int | None=None, region: Region | np.ndarray | list | None=None,
              target_fields=None, exact_cut=True, n_workers=config['DEFAULT_N_PROCS'],
              use_process=True, copy_result=True, use_vname_mapping=True):
    """
    Read dark matter particle data from HDF5 file.
    """
    return read_part(path, 'dm', iout=iout, region=region, target_fields=target_fields, 
                     exact_cut=exact_cut, n_workers=n_workers, use_process=use_process, 
                     copy_result=copy_result, use_vname_mapping=use_vname_mapping)

def read_sink(path: str, iout: int | None=None, region: Region | np.ndarray | list | None=None,
              target_fields=None, exact_cut=True, n_workers=config['DEFAULT_N_PROCS'],
              use_process=True, copy_result=True, use_vname_mapping=True):
    """
    Read sink particle data from HDF5 file.
    """
    return read_part(path, 'sink', iout=iout, region=region, target_fields=target_fields, 
                     exact_cut=exact_cut, n_workers=n_workers, use_process=use_process, 
                     copy_result=copy_result, use_vname_mapping=use_vname_mapping)

def read_tracer(path: str, iout: int | None=None, region: Region | np.ndarray | list | None=None,
              target_fields=None, exact_cut=True, n_workers=config['DEFAULT_N_PROCS'],
              use_process=True, copy_result=True, use_vname_mapping=True):
    """
    Read tracer particle data from HDF5 file.
    """
    return read_part(path, 'tracer', iout=iout, region=region, target_fields=target_fields, 
                     exact_cut=exact_cut, n_workers=n_workers, use_process=use_process, 
                     copy_result=copy_result, use_vname_mapping=use_vname_mapping)


def get_info(path: str, iout: int, cosmo=True, cosmo_table=None) -> dict:
    """
    Get simulation info from HDF5 file attributes.

    Parameters
    ----------
    path : str
        Path to the directory containing HDF5 files.
    iout : int
        Output number to construct the filename.
    cosmo : bool, optional
        Whether to include cosmology table and lookback time. Defaults to True.
    cosmo_table : dict, optional
        Precomputed cosmology table. If None, it will be created from file attributes.
    """
    filenames = [os.path.join(path, config['FILENAME_FORMAT_HDF'].format(data=data, iout=iout)) for data in ['cell', 'part']]
    filenames = [fn for fn in filenames if os.path.exists(fn)]
    if len(filenames) == 0:
        raise FileNotFoundError(f"No HDF5 files found for iout={iout} in {path}")
    filename = filenames[0]
    
    with h5py.File(filename, 'r') as f:
        attrs = dict(f.attrs)
        if cosmo:
            if cosmo_table is None:
                H0 = attrs.get('H0', 70.0)
                omega_m = attrs.get('omega_m', 0.3)
                omega_l = attrs.get('omega_l', 0.7)
                omega_k = attrs.get('omega_k', 0.0)
                omega_r = attrs.get('omega_r', 0.0)
                cosmo_table = get_cosmo_table(H0, omega_m, omega_l, omega_k=omega_k, omega_r=omega_r)
            attrs['cosmo_table'] = cosmo_table
            attrs['lookback_time'] = cosmo_convert(attrs['cosmo_table'], 1.0, 'aexp', 'age') / cgs_unit.Gyr - attrs['age']

    return attrs


def repack(path, path_new):
    def copy_attrs(src, dst):
        """Recursively copy all attributes"""
        for key in src.attrs.keys():
            dst.attrs[key] = src.attrs[key]
    
    def copy_group(src_group, dst_group):
        """Recursively copy all groups, datasets, and attributes"""
        copy_attrs(src_group, dst_group)
        
        for name, item in src_group.items():
            if isinstance(item, h5py.Group):
                # Create group and recursively copy its contents
                new_group = dst_group.create_group(name)
                copy_group(item, new_group)
            elif isinstance(item, h5py.Dataset):
                # Copy dataset with attributes
                dst_group.copy(item, name)
    
    with h5py.File(path, "r") as f_old, h5py.File(path_new, "w") as f_new:
        copy_group(f_old, f_new)