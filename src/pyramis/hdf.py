import os
import h5py
import numpy as np
import glob
from typing import Sequence

import warnings

from .config_module import get_mapping, get_vname

from . import get_config, get_dim_keys, cgs_unit, timer, get_position_keys
from .core import compute_chunk_list_from_hilbert
from .geometry import Region, Box
from .utils.arrayview import ArrayView
from .utils import run_mp_executor
from .utils.hilbert import HILBERT_KEY_DTYPE, hilbert_to_compound
from. import ramses
from .astro import get_cosmo_table, cosmo_convert
from . import ANY
from .ramses import _scheduled_snapshots

from multiprocessing.shared_memory import SharedMemory


def check_snapshots(path: str, check_data=['cell', 'part'], check_info=['aexp', 'age', 'time', 'icoarse', 'scheduled'], report_missing=False, scale_threshold=50.) -> np.ndarray:
    config = get_config()
    timer.start(f"Checking HDF snapshots at {path} for {check_data}...")
    iout_list = None
    if isinstance(check_data, str):
        check_data = [check_data]
    if len(check_data) == 0:
        raise ValueError("check_data cannot be empty. Please specify at least one of 'cell' or 'part'.")
    for data in check_data:
        pattern = config['FILENAME_FORMAT_HDF'].format(data=data, iout=ANY)
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
    
    if 'scheduled' in check_info and 'icoarse' not in check_info:
        check_info += ['icoarse']
    
    info_list = [[] for _ in check_info]
    iout_list_new = []
    aout, tout = None, None
    if len(check_info) > 0:
        for iout in iout_list:
            try:
                info = read_info(path, iout, cosmo=False, check_data=check_data)
            except BlockingIOError:
                timer.message(f"Skipping file for iout={iout}, which is currently locked.")
                continue
            except OSError:
                timer.message(f"Skipping file for iout={iout}, which cannot be read.")
                continue
            iout_list_new.append(iout)
            for idx, key in enumerate(check_info):
                if key == 'scheduled':
                    info_list[idx].append(False)
                else:
                    info_list[idx].append(info.get(key))
            aout = info.get('aout', aout)
            tout = info.get('tout', tout)
        info_list = [np.array(lst) for lst in info_list]
        
        table = np.rec.fromarrays(
            [iout_list_new, *info_list],
            dtype=[('iout', 'i4'), *[(key, info_list[idx].dtype) for idx, key in enumerate(check_info)]])
    else:
        table = np.rec.fromarrays(
            [iout_list], dtype=[('iout', 'i4')])
    table = np.sort(table, order='iout')

    if 'scheduled' in check_info:
        scheduled = np.zeros(len(table), dtype=bool)
        scheduled[table['iout'] == 1] = True # always include the first snapshot

        if 'aexp' in check_info and aout is not None and len(aout) > 0 and not np.all(aout == 0.0):
            a_thr = table['aexp'] / table['icoarse'] * scale_threshold
            scheduled |= _scheduled_snapshots(aout, table['aexp'], a_thr, iout=table['iout'], report_missing=report_missing)

        if 'time' in check_info and tout is not None and len(tout) > 0 and not np.all(tout == 0.0):
            t_thr = table['time'] / table['icoarse'] * scale_threshold
            scheduled |= _scheduled_snapshots(tout, table['time'], t_thr, iout=table['iout'], report_missing=report_missing)

        table['scheduled'] = scheduled

    timer.record(f"Found {table.size} snapshots in {path} with data {check_data}.")
    return table


def get_by_type(obj: h5py.File | h5py.Group, name:str, datatype=None):
    data = obj.get(name)
    if datatype is not None:
        assert isinstance(data, datatype), f"{name} is not of type {datatype}"
    return data


def _load_hilbert_boundary(dataset: h5py.Dataset) -> np.ndarray:
    """Load hilbert_boundary as HILBERT_KEY_DTYPE compound array.
    Supports both new compound format [('hi', u64), ('lo', u64)] and legacy float128 format.
    """
    raw = dataset[:]
    if raw.dtype.names is not None and 'hi' in raw.dtype.names:
        return raw  # already compound
    # Legacy: float128 (or float64 on Windows) stored as plain float array
    return hilbert_to_compound(np.array([int(x) for x in raw], dtype=object))


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
    config = get_config()
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


def _load_filter_worker(args):
    path, group_name, target_fields_native, dtype_out, start, end, region, is_cell, subsample = args
    with h5py.File(path, 'r', locking=False) as f:
        group = f.get(group_name)
        data = group.get('data')
        if target_fields_native is not None:
            data = data.fields(target_fields_native)
        if subsample is None:
            subsample = 1
        data_slice = data[start:end:subsample].view(dtype_out)
        boxsize = f.attrs.get('boxsize', 1.0)
        mask = region.contains_data(data_slice, cell=is_cell, boxlen=boxsize)
        data_slice = data_slice[mask]
        return data_slice


def _load_slice_worker(args):
    (path, group_name, target_fields_native,
     shm_name, shared_arr, ndata_tot, dtype_out,
     start, end, offset, ndata) = args

    with h5py.File(path, 'r', locking=False) as f:
        group = f.get(group_name)
        data = group.get('data')
        if target_fields_native is not None:
            data = data.fields(target_fields_native)
        data_slice = data[start:end].view(dtype_out)

        if shm_name is not None:
            shm = SharedMemory(name=shm_name)
            try:
                target = np.ndarray((ndata_tot,), dtype=dtype_out, buffer=shm.buf)
                target[offset:offset + ndata] = data_slice
            finally:
                shm.close()
        else:
            shared_arr[offset:offset + ndata] = data_slice

def _chunk_slice_hdf_mp(
    path,
    group_name,
    starts,
    ends,
    dtype_out,
    target_fields,
    region: Region | None=None,
    n_workers=None,
    mp_backend="process",
    copy_result=True,
    is_cell=False,
    subsample=None):
    config = get_config()

    if n_workers is None:
        n_workers = config['DEFAULT_N_PROCS']

    starts = np.asarray(starts)
    ends = np.asarray(ends)

    if region is not None:
        # 1-pass: each worker reads, filters, and returns its chunk directly
        jobs = [
            ((path, group_name, target_fields, dtype_out, int(start), int(end), region, is_cell, subsample),)
            for start, end in zip(starts, ends)
        ]
        chunks = run_mp_executor(_load_filter_worker, jobs, backend=mp_backend, n_workers=n_workers, mp_method='submit')
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=dtype_out)

    ndata_per_chunk = ends - starts
    ndata_tot = int(np.sum(ndata_per_chunk))

    if ndata_tot == 0:
        return np.empty((0,), dtype=dtype_out)

    offsets = np.zeros_like(ndata_per_chunk)
    offsets[1:] = np.cumsum(ndata_per_chunk[:-1])
    offsets = offsets.astype(int)

    itemsize = dtype_out.itemsize
    total_bytes = ndata_tot * itemsize

    if mp_backend == "process" and n_workers > 1:
        shm = SharedMemory(create=True, size=total_bytes)
        try:
            shared_arr = np.ndarray((ndata_tot, ), dtype=dtype_out, buffer=shm.buf)
            jobs = [
                ((path, group_name, target_fields, shm.name, None, ndata_tot, dtype_out, int(start), int(end), int(offset), int(ndata)),)
                for start, end, offset, ndata in zip(starts, ends, offsets, ndata_per_chunk)
                if ndata > 0]
            run_mp_executor(_load_slice_worker, jobs, backend=mp_backend, n_workers=n_workers, mp_method='submit')
            if copy_result:
                result = np.array(shared_arr, copy=True)
            else:
                result = ArrayView(shm, (ndata_tot,), dtype_out)
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
            ((path, group_name, target_fields, None, shared_arr, ndata_tot, dtype_out, int(start), int(end), int(offset), int(size)),)
            for start, end, offset, size in zip(starts, ends, offsets, ndata_per_chunk)
            if size > 0]
        run_mp_executor(_load_slice_worker, jobs, backend=mp_backend, n_workers=n_workers, mp_method='submit')
        result = shared_arr

    return result

def _prepare_hdf_read(f, group_name, boundary_name, chunk_indices, chunk_sizes, target_fields, vname_set, use_vname_mapping):
    vname_set_file = f.attrs.get('vname_set', 'native')
    if vname_set_file == vname_set:
        use_vname_mapping = False
    group = get_by_type(f, group_name, h5py.Group)
    data = get_by_type(group, 'data', h5py.Dataset)
    bounds = get_by_type(group, boundary_name, h5py.Dataset)
    starts, ends = bounds[chunk_indices], bounds[chunk_indices + chunk_sizes]

    dtype_file = data.dtype
    mapping = get_mapping(vname_set_file, vname_set)
    if target_fields is not None:
        if use_vname_mapping:
            mapping_reverse = get_mapping(vname_set, vname_set_file)
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
        filename: str, 
        name: str, 
        region: Region | np.ndarray | list | None = None, 
        target_fields: list | None = None, 
        levelmax: int | None = None,
        levelmin: int | None = None,
        exact_cut: bool = True,
        n_workers: int | None = None,
        use_process: bool = True,
        copy_result: bool = True,
        is_cell: bool = False,
        vname_set: str | None = None,
        use_vname_mapping: bool = True,
        subsample: int = 1) -> ArrayView:

    config = get_config()

    timer.start(f"Reading HDF5 data from {filename} in group {name}...")

    if n_workers is None:
        n_workers = config['DEFAULT_N_PROCS']

    if vname_set is None:
        vname_set = config['VNAME_SET']

    if use_process:
        mp_backend = "process"
    else:
        mp_backend = "thread"
    
    if exact_cut and region is not None and target_fields is not None:
        warn = False
        pos_keys = get_position_keys()
        for key in pos_keys:
            if key not in target_fields:
                warn = True
                target_fields = list(target_fields) + [key]

        vname_level = get_vname('level')
        if vname_level not in target_fields and is_cell:
            warn = True
            target_fields = list(target_fields) + [vname_level]

        if warn:
            warnings.warn("Exact cut with region specified requires position fields to be loaded. They have been added to target_fields.")

    if isinstance(region, np.ndarray) or isinstance(region, list):
        region = Box(region)

    region_cut = region if exact_cut else None

    with h5py.File(filename, 'r') as f:
        info = read_info_from_hdf(f)
        group = get_by_type(f, name, h5py.Group)
        nchunks = int(group.attrs.get('n_chunk', 0))
        if region is not None:
            chunk_indices = compute_chunk_list_from_hilbert(
                region=region,
                hilbert_boundary=_load_hilbert_boundary(get_by_type(group, 'hilbert_boundary', h5py.Dataset)),
                level_hilbert=group.attrs.get('levelmax', 1),
                boxlen=group.parent.attrs.get('boxlen', 1.0),
                n_workers=n_workers
            )
        else:
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

        timer.message(f"Total number of chunks to read: {len(chunk_indices)} / {nchunks}.")

        data, dtype_out, target_fields_native, starts, ends = _prepare_hdf_read(
            f, name, 'chunk_boundary', chunk_indices, chunk_sizes,
            target_fields, vname_set, use_vname_mapping)
        boxsize = f.attrs.get('boxsize', 1.0)

        if n_workers == 1:
            if region_cut is None:
                # zero-copy path: pre-allocate and fill in-place
                ndata_per_chunk = ends - starts
                ndata_tot = int(np.sum(ndata_per_chunk))
                result = np.empty(ndata_tot, dtype=dtype_out)
                offset = 0
                for start, end, ndata in zip(starts, ends, ndata_per_chunk):
                    result[offset:offset + ndata] = data[start:end].view(dtype_out)
                    offset += ndata
            else:
                output_list = []
                for start, end in zip(starts, ends):
                    data_slice = data[start:end:subsample].view(dtype_out)
                    mask = region_cut.contains_data(data_slice, cell=is_cell, boxlen=boxsize)
                    output_list.append(data_slice[mask])
                result = np.concatenate(output_list) if output_list else np.empty(0, dtype=dtype_out)

    if n_workers > 1:
        result = _chunk_slice_hdf_mp(
            filename, name, starts, ends, dtype_out, target_fields_native,
            region=region_cut, n_workers=n_workers, mp_backend=mp_backend,
            copy_result=copy_result, is_cell=is_cell, subsample=subsample)

    timer.record(f"Finished reading HDF5 data from {filename}. Found {len(result)} items.")
    if isinstance(result, ArrayView):
        result.info = info
    else:
        result = ArrayView(result, info=info)

    return result


def read_part(
        path: str,
        iout: int | None=None,
        region: Region | np.ndarray | list | None=None,
        target_fields=None,
        part_type: str | None=None,
        exact_cut=True,
        n_workers=None,
        use_process=True,
        copy_result=True,
        vname_set=None,
        use_vname_mapping=True,
        subsample=1) -> ArrayView:
    """
    Read particle data from HDF5 file.

    Parameters
    ----------
    path : str
        Path to the directory containing HDF5 files or full file path if iout is None.
    iout : int, optional
        Output number to construct the filename. If None, `path` is treated as the full filename.
    region : Region or array-like, optional
        Region to filter particles. If None, all particles are read.
    target_fields : list, optional
        List of fields to read. If None, all fields are read.
    part_type : str, optional
        Type of particles to read (e.g., 'star', 'dm', etc.). If None, all particle types
        found in the file are read and concatenated. When types have different fields, only
        the common fields are kept.
    exact_cut : bool, optional
        Whether to apply exact cut based on the region. Defaults to True.
    n_workers : int, optional
        Number of parallel workers to use. Defaults to config['DEFAULT_N_PROCS'].
    use_process : bool, optional
        Whether to use process-based parallelism. Defaults to True.
    copy_result : bool, optional
        Whether to return a copy of the result array. Defaults to True.
    use_vname_mapping : bool, optional
        Whether to apply variable name mapping. Defaults to True.
    subsample : int, optional
        Subsampling factor to apply when reading data. If >1, only every nth cell is read. Defaults to 1 (no subsampling).

    Returns
    -------
    ArrayView
        Array of particle data.
    """

    config = get_config()

    if vname_set is None:
        vname_set = config['VNAME_SET']

    if n_workers is None:
        n_workers = config['DEFAULT_N_PROCS']

    if iout is None:
        filename = path
    else:
        filename = os.path.join(path, config['FILENAME_FORMAT_HDF'].format(data='part', iout=iout))

    if part_type is None:
        with h5py.File(filename, 'r') as f:
            info = read_info_from_hdf(f)
            part_types = [k for k, v in f.items() if isinstance(v, h5py.Group) and 'n_chunk' in v.attrs]
        if not part_types:
            return ArrayView(np.empty(0, dtype=np.float64), info=info)
        arrays = [
            read_hdf(filename, pt, region=region, target_fields=target_fields, exact_cut=exact_cut,
                     n_workers=n_workers, use_process=use_process, copy_result=copy_result,
                     is_cell=False, vname_set=vname_set, use_vname_mapping=use_vname_mapping)
            for pt in part_types
        ]
        dtypes = [a.dtype for a in arrays]
        if len(set(dtypes)) == 1:
            return ArrayView(np.concatenate(arrays), info=info)
        common = [f for f in dtypes[0].names if all(f in dt.names for dt in dtypes[1:])]
        common_dtype = np.dtype([(f, dtypes[0][f]) for f in common])
        parts = []
        for arr in arrays:
            out = np.empty(len(arr), dtype=common_dtype)
            for f in common:
                out[f] = arr[f]
            parts.append(out)
        return ArrayView(np.concatenate(parts), info=info)

    return read_hdf(filename, part_type, region=region, target_fields=target_fields, exact_cut=exact_cut, n_workers=n_workers, use_process=use_process, copy_result=copy_result, is_cell=False, vname_set=vname_set, use_vname_mapping=use_vname_mapping, subsample=subsample)


def read_cell(
        path: str,
        iout: int | None=None,
        region: Region | np.ndarray | list | None=None,
        target_fields=None,
        levelmax_load=None,
        exact_cut=True,
        n_workers=None,
        use_process=True,
        copy_result=True,
        read_branch=False,
        vname_set=None,
        use_vname_mapping=True,
        subsample=1) -> ArrayView:
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
    subsample : int, optional
        Subsampling factor to apply when reading data. If >1, only every nth cell is read. Defaults to 1 (no subsampling).

    Returns
    -------
    np.ndarray
        Array of cell data.
    """
    
    config = get_config()

    if vname_set is None:
        vname_set = config['VNAME_SET']

    if n_workers is None:
        n_workers = config['DEFAULT_N_PROCS']
    
    if iout is None:
        filename = path
    else:
        filename = os.path.join(path, config['FILENAME_FORMAT_HDF'].format(data='cell', iout=iout))
    if levelmax_load is not None:
        data_leaf = read_hdf(filename, 'branch', region=region, target_fields=target_fields, exact_cut=exact_cut, n_workers=n_workers, levelmax=levelmax_load, use_process=use_process, copy_result=copy_result, is_cell=True, vname_set=vname_set, use_vname_mapping=use_vname_mapping, subsample=subsample)
        data_branch = read_hdf(filename, 'leaf', region=region, target_fields=target_fields, exact_cut=exact_cut, n_workers=n_workers, levelmin=levelmax_load, levelmax=levelmax_load, use_process=use_process, copy_result=copy_result, is_cell=True, vname_set=vname_set, use_vname_mapping=use_vname_mapping, subsample=subsample)
        data = np.concatenate([data_leaf, data_branch])
    elif read_branch:
        data = read_hdf(filename, 'branch', region=region, target_fields=target_fields, exact_cut=exact_cut, n_workers=n_workers, use_process=use_process, copy_result=copy_result, is_cell=True, vname_set=vname_set, use_vname_mapping=use_vname_mapping, subsample=subsample)
    else:
        data = read_hdf(filename, 'leaf', region=region, target_fields=target_fields, exact_cut=exact_cut, n_workers=n_workers, use_process=use_process, copy_result=copy_result, is_cell=True, vname_set=vname_set, use_vname_mapping=use_vname_mapping, subsample=subsample)
    return data


def _generate_part_reader(part_type: str):
    def _reader(
        path: str,
        iout: int | None = None,
        region: Region | np.ndarray | list | None = None,
        target_fields=None,
        exact_cut: bool = True,
        n_workers=None,
        use_process: bool = True,
        copy_result: bool = True,
        vname_set=None,
        use_vname_mapping: bool = True,
    ):
        return read_part(
            path,
            iout=iout,
            region=region,
            target_fields=target_fields,
            part_type=part_type,
            exact_cut=exact_cut,
            n_workers=n_workers,
            use_process=use_process,
            copy_result=copy_result,
            vname_set=vname_set,
            use_vname_mapping=use_vname_mapping,
        )

    return _reader

read_star = _generate_part_reader("star")
read_dm = _generate_part_reader("dm")
read_sink = _generate_part_reader("sink")
read_tracer = _generate_part_reader("tracer")


def read_sinkprops(
        path: str,
        filename='SINKPROPS/sinkprops.h5',
        target_id: int | Sequence[int] | np.ndarray=None,
        icoarse_min: int | None=None,
        icoarse_max:int | None=None,
        return_data=True,
        return_sinks=False,
        return_steps=False,
        target_fields=None,
        vname_set=None,
        use_vname_mapping=True):

    config = get_config()
    filename = os.path.join(path, filename)
    timer.start(f"Reading sink properties from {filename}...")

    if vname_set is None:
        vname_set = config['VNAME_SET']

    with h5py.File(filename, 'r') as f:
        if return_data:
            data = get_by_type(f, 'data', h5py.Dataset)
            dtype_file = data.dtype

            if icoarse_max is not None and icoarse_max < 0:
                icoarse_max = f.attrs.get('icoarse_max', 0) + icoarse_max + 1
            if icoarse_min is not None and icoarse_min < 0:
                icoarse_min = f.attrs.get('icoarse_max', 0) + icoarse_min + 1

            vname_set_file = f.attrs.get('vname_set', 'native')
            mapping = get_mapping(vname_set_file, vname_set)
            mapping_reverse = None
            if target_fields is not None:
                mapping_reverse = get_mapping(vname_set, vname_set_file)
                target_fields_file = [mapping_reverse.get(f, f) for f in target_fields if mapping_reverse.get(f, f) in dtype_file.names]
            else:
                target_fields_file = None
            dtype_out = np.dtype([(name, dtype_file.fields[name][0]) for name in target_fields_file]) if target_fields_file is not None else dtype_file
            dtype_out = remap_dtype_names(dtype_out, mapping) if use_vname_mapping else dtype_out

            if return_sinks or target_id is not None:
                sinks = get_by_type(f, 'sinks', h5py.Dataset)
            
            if return_steps or icoarse_max is not None or icoarse_min is not None:
                steps = get_by_type(f, 'steps', h5py.Dataset)

            if target_id is not None:
                if mapping_reverse is None:
                    mapping_reverse = get_mapping(vname_set, vname_set_file)
                id_field_name = mapping_reverse.get('identity', 'identity') if use_vname_mapping else 'identity'
                if id_field_name not in sinks.dtype.names:
                    raise ValueError(f"ID field '{id_field_name}' not found in sinkprops dataset.")
                id_data = sinks[id_field_name]
                if np.isscalar(target_id):
                    target_id_set = np.array([target_id])
                else:
                    target_id_set = np.unique(target_id)
                sinks = sinks[np.isin(id_data, target_id_set)]
                offsets = sinks['offset']
                sizes = sinks['num']

                size_total = np.sum(sizes)
                data_array = np.empty(size_total, dtype=dtype_out)
                start = 0
                for offset, size in zip(offsets, sizes):
                    data_slice = data[offset:offset+size]
                    if target_fields_file is not None:
                        data_slice = data_slice.fields(target_fields_file)
                    data_array[start:start+size] = data_slice
                    start += size
                if icoarse_max is not None or icoarse_min is not None:
                    mask = (data_array['icoarse'] <= (icoarse_max if icoarse_max is not None else np.inf)) & (data_array['icoarse'] >= (icoarse_min if icoarse_min is not None else 1))
                    data_array = data_array[mask]

            else:
                if icoarse_max is not None or icoarse_min is not None:
                    mask = (steps['icoarse'] <= (icoarse_max if icoarse_max is not None else np.inf)) \
                            & (steps['icoarse'] >= (icoarse_min if icoarse_min is not None else 1))
                    steps_target = steps[mask]
                    offsets = steps_target['offset']
                    sizes = steps_target['num']
                    icoarse_key = get_by_type(f, 'icoarse_key', h5py.Dataset)
                    key_array = np.empty(np.sum(sizes), dtype=icoarse_key.dtype)
                    start = 0
                    for offset, size in zip(offsets, sizes):
                        key_array[start:start+size] = icoarse_key[offset:offset+size]
                        start += size
                    key_array = np.sort(key_array)
                    data_array = data[key_array]
                else:
                    data_array = data
            data_array = data_array[:]

            out = data_array.view(dtype_out)
        else:
            out = None
        if return_sinks:
            dtype_sinks = remap_dtype_names(sinks.dtype, mapping) if use_vname_mapping else sinks.dtype
            sinks = sinks[:].view(dtype_sinks)
            if out is not None:
                out = (out, sinks)
            else:
                out = sinks

        if return_steps:
            dtype_steps = remap_dtype_names(steps.dtype, mapping) if use_vname_mapping else steps.dtype
            steps = steps[:].view(dtype_steps)
            if out is not None:
                out = (out, steps) if not return_sinks else (out, sinks, steps)
            else:
                out = steps
    timer.record(f"Finished reading sink properties from {filename}. Found {len(data_array)} items.")

    return out


def read_info_from_hdf(f: h5py.File, cosmo=True, cosmo_table=None) -> dict:
    """
    Extract simulation info from an open HDF5 file.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file object.
    cosmo : bool, optional
        Whether to include cosmology table and lookback time. Defaults to True.
    cosmo_table : dict, optional
        Precomputed cosmology table. If None, it will be created from file attributes.

    Returns
    -------
    dict
        Dictionary containing simulation info extracted from file attributes.
    """
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
        attrs['lookback_time'] = cosmo_convert(attrs['cosmo_table'], 1.0, 'aexp', 'age') / cgs_unit['Gyr']['factor'] - attrs['age']
    return attrs


def read_info(path: str, iout: int, cosmo=True, cosmo_table=None, check_data=['cell', 'part']) -> dict:
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

    config = get_config()

    filenames = [os.path.join(path, config['FILENAME_FORMAT_HDF'].format(data=data, iout=iout)) for data in check_data]
    filenames = [fn for fn in filenames if os.path.exists(fn)]
    if len(filenames) == 0:
        raise FileNotFoundError(f"No HDF5 files found for iout={iout} in {path}")

    attrs = None
    for fn in filenames:
        timer.message(f"Reading simulation info from {fn}...", 2)
        try:
            with h5py.File(fn, 'r') as f:
                attrs = read_info_from_hdf(f, cosmo=cosmo, cosmo_table=cosmo_table)
                break
        except BlockingIOError:
            timer.message(f"Skipping file {fn}, which is currently locked.")
            continue
    if attrs is None:
        raise BlockingIOError(f"Could not read attributes from any file for iout={iout} in {path}")

    return attrs


def get_ndata(path, data, iout, data_type):
    config = get_config()
    filename = os.path.join(path, config['FILENAME_FORMAT_HDF'].format(data=data, iout=iout))
    with h5py.File(filename, 'r') as f:
        if data_type in f.keys():
            group = get_by_type(f, data_type, h5py.Group)
            return group.attrs['size']
        else:
            return 0


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