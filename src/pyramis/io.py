import os
import glob
import h5py
import numpy as np
from typing import Callable, Sequence, overload, Literal, Tuple
import warnings

from concurrent.futures import as_completed
import configparser

import re
from . import get_config, get_dim_keys, get_position, get_velocity, get_cosmo_table, cosmo_convert, get_vname, get_cell_size
from .core import compute_chunk_list_from_hilbert, str_to_tuple, quad_to_f16
from pyramis.geometry import Region, Box
from .utils.fortranfile import FortranFile
from .utils.arrayview import SharedView
from .utils import get_mp_executor

import numpy as np
import h5py
from multiprocessing.shared_memory import SharedMemory
from itertools import repeat

config = get_config()

def get_available_snapshots(path: str, check_data=['amr', 'hydro', 'part'], report_missing=False, scheduled_only=False, namelist_path=None, scale_threshold=5.) -> np.ndarray:
    pattern = os.path.join(path, config['OUTPUT_FORMAT_ANY'])
    dirs = glob.glob(pattern)
    iout_list = []
    aexp_list = []
    time_list = []
    nstep_coarse_list = []
    for d in dirs:
        basename = os.path.basename(d)
        ok = True

        iout = int(basename.split('_')[-1])
        info_path = os.path.join(d, f'info_{iout:05d}.txt')

        if os.path.exists(info_path) is False:
            if report_missing:
                warnings.warn(f"Info file missing for iout={iout} in directory {d}.", UserWarning)
            continue
            
        info = parse_info(info_path)

        for data in check_data:
            file_pattern = os.path.join(d, config['FILENAME_FORMAT_ANY'].format(data=data, iout=iout))
            files = glob.glob(file_pattern)
            if len(files) != info['ncpu']:
                if report_missing:
                    warnings.warn(f"Number of '{data}' files does not match for iout={iout}. Expected {info['ncpu']} files, found {len(files)}.", UserWarning)
                ok = False                    
        if ok:
            iout_list.append(iout)
            aexp_list.append(info['aexp'])
            time_list.append(info['time'])
            nstep_coarse_list.append(info['nstep_coarse'])
    
    table = np.rec.fromarrays([iout_list, aexp_list, time_list, nstep_coarse_list], dtype=[('iout', 'i4'), ('aexp', 'f8'), ('time', 'f8'), ('nstep_coarse', 'i4')])
    table.sort(order='iout')

    if scheduled_only:
        # get the latest snapshot info
        iout_check = table['iout'][-1]
        info = get_info(path, iout_check)

        aout, tout = [], []
        if 'aout' in info:
            aout = np.array(info['aout'])
        if 'tout' in info:
            tout = np.array(info['tout'])

        if 'aout' not in info and 'tout' not in info:
            if namelist_path is None:
                namelist_path = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout_check), config['NAMELIST_FILENAME'])
            nml = parse_namelist(namelist_path)

            scheduled = np.zeros(len(table), dtype=bool)

            if 'aout' in nml:
                aout = np.array(tuple(aout))
            if 'tout' in nml:
                tout = np.array(tuple(tout))

        if aout.size > 0:
            a_thr = np.max(aout) / info['nstep_coarse'] * scale_threshold
        if tout.size > 0:
            t_thr = np.max(tout) / info['nstep_coarse'] * scale_threshold

        scheduled = np.zeros(len(table), dtype=bool)
        scheduled[0] = True # always include the first snapshot
        for a in aout:
            # find the closest aexp in the table, except those already found
            diff = np.abs(table['aexp'] - a)
            diff_masked = np.where(scheduled, np.inf, diff)
            cand_key = np.argmin(diff_masked)
            cand = table[cand_key]
            if np.abs(cand['aexp'] - a) < a_thr:
                scheduled[cand_key] = True
        
        for t in tout:
            diff = np.abs(table['time'] - t)
            diff_masked = np.where(scheduled, np.inf, diff)
            cand_key = np.argmin(diff_masked)
            cand = table[cand_key]
            if np.abs(cand['time'] - t) < t_thr:
                scheduled[cand_key] = True

        table = table[scheduled]

    return table


def read_type_descriptor(path: str, iout: int, data: str='part') -> np.dtype:
    fd_path = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout), config['FILE_DESCRIPTOR_FORMAT'].format(data=data))
    if not os.path.exists(fd_path):
        raise FileNotFoundError(f"File descriptor not found: {fd_path}")
    fd = np.genfromtxt(fd_path, delimiter=",", names=True, dtype=None, encoding="utf-8", skip_header=1, autostrip=True)
    data_dtype = []
    for p in fd:
        vname = get_vname(p['variable_name'])
        data_dtype.append((str(vname), p['variable_type']))
    return np.dtype(data_dtype)


def parse_info(path):
    data = {}
    pattern = re.compile(r"""
        ^\s*
        (?P<key>[A-Za-z0-9_ ]+?)       # key (can have spaces)
        \s*=\s*
        (?P<val>[-+0-9.eE]+|[A-Za-z_]+)
        \s*$
    """, re.VERBOSE)

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip blanks
            m = pattern.match(line)
            if not m:
                continue

            key = m.group("key").strip().replace(" ", "_")
            val = m.group("val")

            # Try to parse numeric value (int → float fallback → string)
            parsed = val
            try:
                parsed = int(val)
            except ValueError:
                try:
                    parsed = float(val)
                except ValueError:
                    pass
            data[key] = parsed
    return data


def parse_namelist(filename):
    config = configparser.ConfigParser(allow_no_value=True)
    with open(filename, 'r') as file:
        lines = []
        for line in file:
            # Replace "&groupname" with "[groupname]"
            if line.strip().startswith("&"):
                line = "[" + line.strip()[1:] + "]\n"
            # Skip the "/" end group notation
            elif line.strip() == "/":
                continue
            lines.append(line)
        
        # Parse the adapted config
        config.read_string("".join(lines))
    
    # Convert config to dictionary format
    namelist_data = {s: dict(config.items(s)) for s in config.sections()}
    return namelist_data


def get_info(output_path, iout, namelist_path=None, cosmo=True, cosmo_table=None, read_amr=True, read_hydro=True) -> dict:
    info = parse_info(os.path.join(output_path, config['OUTPUT_FORMAT'].format(iout=iout), f'info_{iout:05d}.txt'))
    info['iout'] = iout

    if read_amr:
        amr_files = glob.glob(os.path.join(output_path, config['OUTPUT_FORMAT'].format(iout=iout), config['FILENAME_FORMAT_ANY'].format(data='amr', iout=iout)))
        if len(amr_files) == 0:
            raise FileNotFoundError(f"No AMR file found at iout = {iout} in {output_path}.")

        amr_path = amr_files[0]
        with FortranFile(amr_path, mode='r') as f:
            info['ncpu'], = f.read_ints()
            info['ndim'], = f.read_ints()
            info['nx'], info['ny'], info['nz'], = f.read_ints()
            info['nlevelmax'], = f.read_ints()
            info['ngridmax'], = f.read_ints()
            info['nboundary'], = f.read_ints()
            info['ngrid_current'] = f.read_ints()
            info['boxlen'], = f.read_reals()

            info['noutput'], info['iout'], info['ifout'], = f.read_ints()
            info['tout'] = f.read_reals()
            info['aout'] = f.read_reals()
            info['time'], = f.read_reals()
            info['dtold'] = f.read_reals()
            info['dtnew'] = f.read_reals()
            info['nstep'], info['nstep_coarse'] = f.read_ints()

            if info['ordering_type'] == 'hilbert':
                f.skip_records(10)
                if info['nboundary'] > 0:
                    f.skip_records(3)
                
                # reads accurate hilbert bounds
                bounds = f.read_record('b')
                if bounds.size == 16 * (info['ncpu'] + 1):
                    # quad case
                    info['bounds'] = quad_to_f16(bounds)
                else:
                    # double case
                    info['bounds'] = bounds.view('f8')

        coarse_min = [0, 0, 0]
        key = ['i', 'j', 'k']
        nxyz = [info['nx'], info['ny'], info['nz']]
        if info['nboundary'] > 0:
            for i in range(info['ndim']):
                if nxyz[i] == 3:
                    coarse_min[i] += 1
                if nxyz[i] == 2:
                    if namelist_path is None:
                        namelist_path = os.path.join(output_path, config['NAMELIST_FILENAME'])
                    nml = parse_namelist(namelist_path)
                    if len(nml) == 0:
                        warnings.warn(f"Assymetric boundaries detected, which cannot be determined without namelist file. \
                                        Move {config['NAMELIST_FILENAME']} file to the output directory or manually apply offset to the cell position.")
                    else:
                        bound_min = np.array(str_to_tuple(nml['BOUNDARY_PARAMS']['%sbound_min' % key[i]]))
                        bound_max = np.array(str_to_tuple(nml['BOUNDARY_PARAMS']['%sbound_max' % key[i]]))
                        if np.any(((bound_min * bound_max) == 1) & bound_min == -1):
                            coarse_min[i] += 1

        # measures x, y, z offset based on the boundary condition
        # does not work if boundaries are asymmetric, which can only be determined in namelist
        info['icoarse_min'] = coarse_min[0]
        info['jcoarse_min'] = coarse_min[1]
        info['kcoarse_min'] = coarse_min[2]

    if read_hydro:
        hydro_files = glob.glob(os.path.join(output_path, config['OUTPUT_FORMAT'].format(iout=iout), config['FILENAME_FORMAT_ANY'].format(data='hydro', iout=iout)))
        if len(hydro_files) > 0:
            hydro_path = hydro_files[0]
            with FortranFile(hydro_path, mode='r') as f:
                f.skip_records(1)
                info['nhvar'], = f.read_ints()
                f.skip_records(3)
                info['gamma'] = f.read_reals()
        else:
            info['nhvar'] = 0

    # build cosmology table if needed
    if cosmo:
        if cosmo_table is not None:
            info['cosmo_table'] = cosmo_table
        else:
            info['cosmo_table'] = get_cosmo_table(
                H0=info['H0'],
                omega_m=info['omega_m'],
                omega_l=info['omega_l'],
                omega_k=info['omega_k'],
                omega_r=info.get('omega_r', None),
            )
        info['age'] = cosmo_convert(info['cosmo_table'], info['aexp'], 'aexp', 'age')
        info['lookback_time'] = cosmo_convert(info['cosmo_table'], 1.0, 'aexp', 'age') - info['age']
        info['z'] = 1.0 / info['aexp'] - 1.0

    return info


def get_data_path(data_name, path, iout, icpu):
    return os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout), config['FILENAME_FORMAT'].format(data=data_name, iout=iout, icpu=icpu))

def _read_npart_file(path, iout, icpu, part_type, family_exists, is_star):
    filename = get_data_path('part', path, iout, icpu)
    with FortranFile(f"{filename}", mode='r') as f:
        # Option 1: read nstar if part_type is 'star'
        f.skip_records(2)
        npart, = f.read_ints('i4')
        
        # Option 2: if part_type is None, just sum npart
        if part_type is None:
            return npart
        f.skip_records(5)

        # Option 3: read family/epoch and classify
        if family_exists:
            # Family-based classification
            data = np.empty(npart, dtype=[('family', np.int8)])
            f.skip_records(9)
            data['family'] = f.read_ints(np.int8)
        else:
            # Parameter-based classification
            data = np.empty(npart, dtype=[('m', np.float64), ('id', np.int32), ('tform', np.float64)])
            f.skip_records(6)
            data['m'] = f.read_reals('f8')
            data['id'] = f.read_ints('i4')
            f.skip_records(1)
            data['tform'] = f.read_reals('f8')
        return np.sum(mask_by_part_type(data, part_type))


def read_npart_header(path, iout):
    filename = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout), f'header_{iout:05d}.txt')
    family_counts = {}

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("particle fields"):
                break

            parts = line.split()
            if len(parts) >= 2:
                family = parts[0].lower()
                count = int(parts[-1])
                family_counts[family] = count
    
    # add tracer count
    tracer_counts = 0
    for key in family_counts.keys():
        if 'tracer' in key:
            tracer_counts += family_counts[key]
    family_counts['tracer'] = tracer_counts

    return family_counts


def read_npart_per_cpu(path, iout, cpulist=None, dtype_read=None, part_type=None, info=None, n_workers: int | None=None, mp_backend: str="thread") -> Sequence[int]:
    if cpulist is None:
        if info is None:
            info = get_info(path, iout)
        cpulist = np.arange(1, int(info['ncpu'])+1)
    if dtype_read is None:
        dtype_read = read_type_descriptor(path, iout, 'part')
    family_exists = dtype_read.names is not None and 'family' in dtype_read.names
    is_star = part_type == 'star'

    npart_cpu = []
    if n_workers in (None, 1):
        for icpu in cpulist:
            npart_cpu.append(_read_npart_file(path, iout, icpu, part_type, family_exists, is_star))
        return npart_cpu
    
    else:
        with get_mp_executor(backend=mp_backend, n_workers=n_workers) as ex:
            results = list(
                ex.map(
                    _read_npart_file,
                    repeat(path),
                    repeat(iout),
                    cpulist,
                    repeat(part_type),
                    repeat(family_exists),
                    repeat(is_star),
                )
            )
        return results


def mask_by_part_type(part, part_type):
    names = part.dtype.names
    if ('family' in names):
        # Do a family-based classification
        mask = np.isin(part['family'], config['PARTICLE_TYPE_DEFINITION'][part_type])
    elif ('tform' in names):
        # Do a parameter-based classification
        if (part_type == 'dm'):
            mask = (part['tform'] == 0) & (part['id'] > 0)
        elif (part_type == 'star'):
            mask = ((part['tform'] < 0) & (part['id'] > 0)) \
                   | ((part['tform'] != 0) & (part['id'] < 0))
        elif (part_type == 'cloud'):
            mask = (part['id'] < 0) & (part['m'] > 0) & (part['tform'] == 0)
        elif (part_type == 'tracer'):
            mask = (part['id'] < 0) & (part['m'] == 0)
        else:
            mask = False
    elif ('id' in names):
        # warnings.warn(f"No `family` or `epoch` field found, using `id` and `mass` instead.", UserWarning)
        # DM-only simulation
        if (part_type == 'dm'):
            mask = part['id'] > 0
        elif (part_type == 'tracer'):
            mask = (part['id'] < 0) & (part['m'] == 0)
        else:
            mask = False
    else:
        # No particle classification is possible
        raise ValueError('Particle data structure not classifiable.')
    return mask


@overload
def read_part(
    path: str, 
    iout: int, 
    region: Region | np.ndarray | list | None = None, 
    cpulist: Sequence[int] | np.ndarray | None = None,
    target_fields: Sequence[str] | None = None,
    part_type: str | None=None, 
    dtype: np.dtype | list | None = None,
    info: dict | None = None,
    read_cpu=False,
    exact_cut: Literal[False] = False,
    n_workers: int=config['DEFAULT_N_PROCS'],
    use_process: Literal[True] = True,
    copy_result: Literal[False] = False) -> SharedView: ...

@overload
def read_part(
    path: str, 
    iout: int, 
    region: Region | np.ndarray | list | None = None, 
    cpulist: Sequence[int] | np.ndarray | None = None,
    target_fields: Sequence[str] | None = None,
    part_type: str | None=None, 
    dtype: np.dtype | list | None = None,
    info: dict | None = None,
    read_cpu=False,
    exact_cut: bool=True,
    n_workers: int=config['DEFAULT_N_PROCS'],
    use_process: bool = False,
    copy_result: bool = True) -> np.ndarray: ...

def read_part(
        path: str, 
        iout: int, 
        region: Region | np.ndarray | list | None = None, 
        cpulist: Sequence[int] | np.ndarray | None = None,
        target_fields: Sequence[str] | None = None,
        part_type: str | None=None, 
        dtype: np.dtype | list | None = None,
        info: dict | None = None,
        read_cpu=False,
        exact_cut: bool=True,
        n_workers: int=config['DEFAULT_N_PROCS'],
        use_process: bool=False,
        copy_result: bool=True) -> np.ndarray | SharedView:

    if isinstance(region, np.ndarray) or isinstance(region, list):
        region = Box(region)

    if info is None:
        info = get_info(path, iout)
    
    if use_process:
        mp_backend = "process"
    else:
        mp_backend = "thread"

    output_dir = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout))
    if dtype is None:
        try:
            dtype = read_type_descriptor(path, iout, 'part')
        except FileNotFoundError as e:
            fd_path = os.path.join(output_dir, config['FILE_DESCRIPTOR_FORMAT'].format(data='part'))
            raise FileNotFoundError(
                f"File descriptor not found: {fd_path}\n"
                f"`dtype` may need to be provided manually.")
    dtype_read = np.dtype(dtype)

    if part_type is not None and target_fields is not None:
        target_fields = list(target_fields)
        if 'family' not in target_fields and 'family' in dtype_read.names:
            # Ensure 'family' is included for classification
            warnings.warn("Including 'family' field for classification.", UserWarning)
            target_fields = target_fields + ['family']
        else:
            if 'tform' not in target_fields and 'tform' in dtype_read.names:
                warnings.warn("Including 'tform' field for classification.", UserWarning)
                target_fields = target_fields + ['tform']
            if 'm' not in target_fields and 'm' in dtype_read.names:
                warnings.warn("Including 'm' field for classification.", UserWarning)
                target_fields = target_fields + ['m']
            if 'id' not in target_fields and 'id' in dtype_read.names:
                warnings.warn("Including 'id' field for classification.", UserWarning)
                target_fields = target_fields + ['id']
    
    dtype_out = dtype_read

    if read_cpu:
        dtype_out = np.dtype(dtype_out.descr + [(get_vname('cpu'), np.int32)])

    if target_fields is not None:
        dtype_out = np.dtype([(name, dtype_out.fields[name][0]) for name in target_fields if name in dtype_out.names])
    if region is not None:
        if cpulist is not None:
            warnings.warn("Both `region` and `cpulist` are provided. `region` will be used to compute `cpulist`.", UserWarning)
        cpulist = compute_chunk_list_from_hilbert(
            region=region,
            hilbert_boundary=info['bounds'],
            level_hilbert=info['nlevelmax']+1,
            boxlen=info['boxlen'],
        ) + 1
    elif cpulist is None:
        cpulist = np.arange(1, int(info['ncpu'])+1)
    else:
        cpulist = np.array(cpulist)

    npart_per_cpu = read_npart_per_cpu(path, iout, cpulist, dtype_read=dtype_read, part_type=part_type)
    npart = np.sum(npart_per_cpu) if len(npart_per_cpu) > 0 else 0
    if npart == 0:
        return np.empty(0, dtype=dtype_out)
    
    args = path, iout, dtype_read, part_type
    
    if n_workers == 1:
        result = _read_cpulist(
            args,
            cpulist,
            dtype_out,
            npart_per_cpu,
            _load_part_file
        )
    else:
        result = _read_cpulist_mp(
            args,
            cpulist,
            dtype_out,
            npart_per_cpu,
            _load_part_file,
            n_workers=n_workers,
            mp_backend=mp_backend,
            copy_result=copy_result
        )
    
    if exact_cut and region is not None:
        result2 = result[region.contains_data(result, cell=False)]
        if isinstance(result, SharedView):
            result.close()
        result = result2

    return result


def _load_part_file(icpu, output_arr, path, iout, dtype_read, part_type=None):

    dtype_out = output_arr.dtype
    filename = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout), config['FILENAME_FORMAT'].format(data='part', iout=iout, icpu=icpu))

    with FortranFile(filename, mode="r") as f:
        f.skip_records(2)
        npart_file, = f.read_ints('i4')
        f.skip_records(5)

        if part_type is None:
            part_data = output_arr
        else:
            # npart_cpu may be larger than npart
            part_data = np.empty(npart_file, dtype=dtype_out)

        for name in dtype_read.names:
            if name not in dtype_out.names:
                f.skip_records(1)
                continue

            dtype_format = dtype_out.fields[name][0]
            if np.issubdtype(dtype_format, np.integer):
                arr = f.read_ints(dtype_format)
            elif np.issubdtype(dtype_format, np.floating):
                arr = f.read_reals(dtype_format)
            else:
                raise TypeError(f"Unsupported data type for field '{name}': {dtype_format}")

            if arr.size != npart_file:
                raise RuntimeError(
                    f"Unexpected size for field '{name}' on CPU {icpu}: "
                    f"got {arr.size}, expected {npart_file}"
                )
            part_data[name] = arr
        
        if get_vname('cpu') in dtype_out.names:
            part_data[get_vname('cpu')] = icpu

        if part_type is not None:
            mask = mask_by_part_type(part_data, part_type)
            part_data = part_data[mask]
            output_arr[:] = part_data


def read_ncell_per_cpu(path, iout, cpulist=None, info=None, read_branch=False) -> Sequence[int]:
    if info is None:
        info = get_info(path, iout)
    if cpulist is None:
        info = get_info(path, iout)
        cpulist = np.arange(1, int(info['ncpu'])+1)

    ndim = info['ndim']
    ncpu = info['ncpu']
    nlevelmax = info['nlevelmax']
    nboundary = info['nboundary']
    twotondim = 2 ** ndim

    ncell_cpu = []
    for icpu in cpulist:
        filename = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout), config['FILENAME_FORMAT'].format(data='amr', iout=iout, icpu=icpu))
        ngridfile = np.empty((nlevelmax, ncpu + nboundary), dtype=np.int32)
        ncell = 0
        with FortranFile(filename, mode='r') as f:
            f.skip_records(21)
            numbl = f.read_ints('i4')
            ngridfile[:, :ncpu] = numbl.reshape((nlevelmax, ncpu))
            f.skip_records(3)
            if nboundary > 0:
                numbb = f.read_ints('i4')
                ngridfile[:, ncpu:] = numbb.reshape((nlevelmax, nboundary))
                f.skip_records(2)

            f.skip_records(4)
            levels, cpus = np.where(ngridfile > 0)
            for ilevel, jcpu in zip(levels, cpus + 1):
                f.skip_records(3)
                if jcpu == icpu:
                    f.skip_records(3 * ndim + 1)
                    for _ in range(twotondim):
                        son = f.read_ints()
                        if not read_branch:
                            if 0 in son:
                                ncell += len(son.flatten()) - np.count_nonzero(son)
                        else:
                            ncell += np.count_nonzero(son)

                    f.skip_records(2 * twotondim)
                else:
                    f.skip_records(3 * (twotondim + ndim) + 1)

        ncell_cpu.append(ncell)
    return ncell_cpu


@overload
def read_cell(
    path: str,
    iout: int,
    region: Region | np.ndarray | list | None = None,
    cpulist: Sequence[int] | np.ndarray | None = None,
    target_fields: Sequence[str] | None = None,
    dtype=None,
    info: dict | None = None,
    read_hydro: bool = True,
    read_grav: bool = False,
    read_cpu: bool = False,
    read_branch: bool = False,
    exact_cut: Literal[False] = False,
    n_workers: int = config['DEFAULT_N_PROCS'],
    use_process: Literal[True] = True,
    copy_result: Literal[False] = False,
) -> SharedView: ...

@overload
def read_cell(
    path: str,
    iout: int,
    region: Region | np.ndarray | list | None = None,
    cpulist: Sequence[int] | np.ndarray | None = None,
    target_fields: Sequence[str] | None = None,
    dtype=None,
    info: dict | None = None,
    read_hydro: bool = True,
    read_grav: bool = False,
    read_cpu: bool = False,
    read_branch: bool = False,
    exact_cut: bool=True,
    n_workers: int = config['DEFAULT_N_PROCS'],
    use_process: bool = False,
    copy_result: bool = True,
) -> np.ndarray: ...

def read_cell(
        path: str, 
        iout: int, 
        region: Region | np.ndarray | list | None = None, 
        cpulist: Sequence[int] | np.ndarray | None = None,
        target_fields: Sequence[str] | None = None,
        dtype=None,
        info: dict | None = None,
        read_hydro=True,
        read_grav=False,
        read_cpu=False,
        read_branch=False,
        exact_cut: bool=True,
        n_workers: int=config['DEFAULT_N_PROCS'],
        use_process: bool=False,
        copy_result: bool=True) -> np.ndarray | SharedView:
    
    if isinstance(region, np.ndarray) or isinstance(region, list):
        region = Box(region)

    if info is None:
        info = get_info(path, iout)

    if use_process:
        mp_backend = "process"
    else:
        mp_backend = "thread"

    output_name = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout))
    fd_path = os.path.join(output_name, config['FILE_DESCRIPTOR_FORMAT'].format(data='hydro'))
    if dtype is None:
        try:
            dtype = read_type_descriptor(path, iout, 'hydro')
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"File descriptor not found: {fd_path}\n"
                f"`dtype` may need to be provided manually. (e.g., [('rho', 'f8'), ('vx', 'f8'), ...])")

    dtype = np.dtype(dtype)

    dim_dtype = [(key, np.float64) for key in get_dim_keys()[:info['ndim']]]
    dtype_out = np.dtype(dim_dtype + [(get_vname('level'), np.int32)])

    if read_hydro:
        dtype_out = np.dtype(dtype_out.descr + dtype.descr)
    
    if read_grav:
        dtype_out = np.dtype(dtype_out.descr + [(get_vname('potential'), np.float64)])
    
    if read_cpu:
        dtype_out = np.dtype(dtype_out.descr + [(get_vname('cpu'), np.int32)])
    
    if target_fields is not None:
        dtype_out = np.dtype([(name, dtype_out.fields[name][0]) for name in target_fields if name in dtype_out.names])
    
    
    if region is not None:
        if cpulist is not None:
            warnings.warn("Both `region` and `cpulist` are provided. `region` will be used to compute `cpulist`.", UserWarning)
        cpulist = compute_chunk_list_from_hilbert(
            region=region,
            hilbert_boundary=info['bounds'],
            level_hilbert=info['nlevelmax']+1,
            boxlen=info['boxlen'],
        ) + 1
    elif cpulist is None:
        cpulist = np.arange(1, int(info['ncpu'])+1)
    else:
        cpulist = np.array(cpulist)
    
    ncell_per_cpu = read_ncell_per_cpu(path, iout, cpulist, info=info, read_branch=read_branch)
    ncell = np.sum(ncell_per_cpu) if len(ncell_per_cpu) > 0 else 0
    if ncell == 0:
        return np.empty(0, dtype=dtype_out)
    
    args = (path, iout, dtype, read_hydro, read_grav, read_branch, info)

    if n_workers == 1:
        result = _read_cpulist(
            args,
            cpulist,
            dtype_out,
            ncell_per_cpu,
            _load_cell_file
        )
    
    else:
        result = _read_cpulist_mp(
            args,
            cpulist,
            dtype_out,
            ncell_per_cpu,
            _load_cell_file,
            n_workers=n_workers,
            mp_backend=mp_backend,
            copy_result=copy_result
        )
    
    if exact_cut and region is not None:
        result2 = result[region.contains_data(result, cell=True, boxsize=info['boxsize'])]
        if isinstance(result, SharedView):
            result.close()
        result = result2
    
    return result

def _load_cell_file(icpu, output_arr, path, iout, dtype_hydro, read_hydro=True, read_grav=False, read_branch=False, info=None):
    OCT_OFFSET = np.array([
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [-0.5,  0.5,  0.5],
        [ 0.5,  0.5,  0.5],
    ])

    if info is None:
        info = get_info(path, iout)
    
    dtype_out = output_arr.dtype
    
    ndim = info['ndim']
    ncpu = info['ncpu']
    nlevelmax = info['nlevelmax']
    nboundary = info['nboundary']
    twotondim = 2 ** ndim
    nhvar = info['nhvar']

    coarse_min = [info['icoarse_min'], info['jcoarse_min'], info['kcoarse_min']]
    boxlen = info['boxlen']

    skip_amr = 3 * (twotondim + ndim) + 1
    ncpu_before = icpu - 1
    ncpu_after = ncpu + nboundary - icpu

    oct_offset_local = OCT_OFFSET[:twotondim, :ndim, np.newaxis]

    mask_hvar = np.isin(dtype_hydro.names, dtype_out.names)

    filename_amr = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout), config['FILENAME_FORMAT'].format(data='amr', iout=iout, icpu=icpu))
    filename_hydro = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout), config['FILENAME_FORMAT'].format(data='hydro', iout=iout, icpu=icpu))
    filename_grav = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout), config['FILENAME_FORMAT'].format(data='grav', iout=iout, icpu=icpu))

    cursor = 0

    ngridfile = np.empty((nlevelmax, ncpu + nboundary), dtype=np.int32)

    jread_arr = np.empty(output_arr.size, dtype=np.uint32)
    nread_arr = np.zeros((nlevelmax, twotondim), dtype=np.uint32)
    cursor = 0

    with FortranFile(filename_amr, mode='r') as f_amr:
        f_amr.skip_records(21)
        numbl = f_amr.read_ints('i4')
        ngridfile[:, :ncpu] = numbl.reshape((nlevelmax, ncpu))
        f_amr.skip_records(3)
        if nboundary > 0:
            numbb = f_amr.read_ints('i4')
            ngridfile[:, ncpu:] = numbb.reshape((nlevelmax, nboundary))
            f_amr.skip_records(2)
        f_amr.skip_records(4)

        for ilevel in range(1, nlevelmax + 1):
            nloop_before = np.count_nonzero(ngridfile[ilevel - 1, :icpu - 1])
            nloop_after = np.count_nonzero(ngridfile[ilevel - 1, icpu:])
            ncache = ngridfile[ilevel - 1, icpu - 1]
            f_amr.skip_records((3 + skip_amr) * nloop_before)
            if ncache > 0:
                f_amr.skip_records(3)

                pos = [] # list of position arrays
                dim_keys = get_dim_keys()
                for idim in range(ndim):
                    if dim_keys[idim] in dtype_out.names:
                        p = f_amr.read_reals(np.float64)
                        pos.append((p + oct_offset_local[:, idim] / 2**ilevel - coarse_min[idim]) * boxlen)
                    else:
                        f_amr.skip_records(1)
                        pos.append(None)
                
                f_amr.skip_records(2 * ndim + 1)
                son = f_amr.read_arrays(twotondim)
                f_amr.skip_records(2 * twotondim)

                ok = (son == 0) if not read_branch else (son != 0)

                # save the indices of the cells to be read
                iread, jread = np.nonzero(ok)
                nread = iread.size
                jread_arr[cursor:cursor + nread] = jread

                # count number of cells read at this level and octant
                nread_arr[ilevel - 1] = np.sum(ok, axis=1)

                for idim in range(ndim):
                    key = get_dim_keys()[idim]
                    if key in dtype_out.names:
                        output_arr[cursor:cursor + nread][key] = pos[idim][iread, jread]
                if 'level' in dtype_out.names:
                    output_arr[cursor:cursor + nread]['level'] = ilevel
                cursor += nread
            f_amr.skip_records((3 + skip_amr) * nloop_after)

    if read_hydro and np.any(mask_hvar):
        skip_hydro = nhvar * twotondim
        cursor = 0
        with FortranFile(filename_hydro, mode='r') as f_hydro:
            f_hydro.skip_records(6)
            for ilevel in range(1, nlevelmax + 1):
                nloop_before = np.count_nonzero(ngridfile[ilevel - 1, :icpu - 1])
                nloop_after = np.count_nonzero(ngridfile[ilevel - 1, icpu:])
                ncache = ngridfile[ilevel - 1, icpu - 1]

                f_hydro.skip_records(2 * ncpu_before + skip_hydro * nloop_before + 2)
                if ncache > 0:
                    for ioct in range(twotondim):
                        nread_oct = nread_arr[ilevel - 1, ioct]
                        jread_oct = jread_arr[cursor:cursor + nread_oct]
                        for ivar, m in enumerate(mask_hvar):
                            if m:
                                var = f_hydro.read_reals()
                                output_arr[cursor:cursor + nread_oct][dtype_hydro.names[ivar]] = var[jread_oct]
                            else:
                                f_hydro.skip_records(1)
                        cursor += nread_oct
                f_hydro.skip_records(2 * ncpu_after + skip_hydro * nloop_after)
    
    if read_grav:
        phi_name = get_vname('potential')
        if phi_name in dtype_out.names:
            cursor = 0
            with FortranFile(filename_grav, mode='r') as f_grav:
                f_grav.skip_records(1)
                ndim1, = f_grav.read_ints()
                f_grav.skip_records(2)

                output_particle_density = ndim1 == ndim + 2
                skip_grav = twotondim * (2 + ndim) if output_particle_density else twotondim * (1 + ndim)

                for ilevel in range(1, nlevelmax + 1):
                    nloop_before = np.count_nonzero(ngridfile[ilevel - 1, :icpu - 1])
                    nloop_after = np.count_nonzero(ngridfile[ilevel - 1, icpu:])
                    ncache = ngridfile[ilevel - 1, icpu - 1]

                    f_grav.skip_records(2 * ncpu_before + skip_grav * nloop_before + 2)
                    if ncache > 0:
                        for ioct in range(twotondim):
                            nread_oct = nread_arr[ilevel - 1, ioct]
                            jread_oct = jread_arr[cursor:cursor + nread_oct]
                            if output_particle_density:
                                f_grav.skip_records(1)
                            if nread_oct > 0:
                                var = f_grav.read_reals()
                                output_arr[cursor:cursor + nread_oct][phi_name] = var[jread_oct]
                                cursor += nread_oct
                                f_grav.skip_records(ndim)
                            else:
                                f_grav.skip_records(1 + ndim)
                    f_grav.skip_records(2 * ncpu_after + skip_grav * nloop_after)


def _read_cpulist(
        args: Tuple,
        cpulist: Sequence[int] | np.ndarray,
        dtype_out: np.dtype,
        ndata_per_cpu: Sequence[int],
        func: Callable) -> np.ndarray:
    ndata = np.sum(ndata_per_cpu) if len(ndata_per_cpu) > 0 else 0
    data = np.empty(ndata, dtype=dtype_out)

    offsets = np.zeros_like(ndata_per_cpu)
    offsets[1:] = np.cumsum(ndata_per_cpu[:-1])
    offsets = offsets.astype(int)

    for icpu, offset, ndata_cpu in zip(cpulist, offsets, ndata_per_cpu):
        if ndata_cpu == 0:
            continue
        func(
            icpu,
            data[offset:offset + ndata_cpu],
            *args
        )
    return data


def _load_data(args: Tuple) -> int:
    (icpu, shm_name, shared_arr, dtype_out, ndata, offset, ndata_cpu, func, *rest) = args

    if ndata_cpu == 0:
        return 0
    # Attach shared memory

    if shm_name is not None:
        shm = SharedMemory(name=shm_name)
        try:
            shared_arr = np.ndarray(
                (ndata,),
                dtype=dtype_out,
                buffer=shm.buf,
            )
            # view for this CPU slice
            output_arr = shared_arr[offset:offset + ndata_cpu]
            func(icpu, output_arr, *rest)
        finally:
            shm.close()
    else:
        output_arr = shared_arr[offset:offset + ndata_cpu]
        func(icpu, output_arr, *rest)

    return ndata_cpu


def _read_cpulist_mp(
        args: Tuple,
        cpulist: Sequence[int] | np.ndarray,
        dtype_out: np.dtype,
        ndata_per_cpu: Sequence[int],
        func: Callable,
        n_workers: int,
        mp_backend: str,
        copy_result: bool) -> np.ndarray | SharedView:
    ndata = np.sum(ndata_per_cpu) if len(ndata_per_cpu) > 0 else 0
    if ndata == 0:
        # No data at all
        empty = np.empty(0, dtype=dtype_out)
        return empty
    
    # Compute offsets in the global array
    offsets = np.zeros_like(ndata_per_cpu)
    offsets[1:] = np.cumsum(ndata_per_cpu[:-1])
    offsets = offsets.astype(int)

    # Allocate shared memory for all data
    itemsize = dtype_out.itemsize
    total_bytes = ndata * itemsize

    if mp_backend == "process":
        shm = SharedMemory(create=True, size=total_bytes)
        try:
            shared_arr = np.ndarray((ndata,), dtype=dtype_out, buffer=shm.buf)
            # Build job list for each CPU
            jobs = [
                (int(icpu), shm.name, None, dtype_out, ndata, int(offset), int(ndata_cpu), func, *args)
                for icpu, offset, ndata_cpu in zip(cpulist, offsets, ndata_per_cpu)
                if ndata_cpu > 0]

            with get_mp_executor(backend=mp_backend, n_workers=n_workers) as executor:
                futures = [executor.submit(_load_data, job) for job in jobs]

                # Propagate the first exception (if any)
                for fut in as_completed(futures):
                    exc = fut.exception()
                    if exc is not None:
                        raise exc

            # At this point, shared_arr is fully populated with all data
            if copy_result:
                result = np.array(shared_arr, copy=True)
            else:
                # Return the shared view; caller must manage shm lifetime
                result = SharedView(shm, (ndata,), dtype_out)
        finally:
            # Clean up shared memory if we own it (copy_result=True).
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
        shared_arr = np.empty((ndata,), dtype=dtype_out)
        # Build job list for each CPU
        jobs = [
            (int(icpu), None, shared_arr, dtype_out, ndata, int(offset), int(ndata_cpu), func, *args)
            for icpu, offset, ndata_cpu in zip(cpulist, offsets, ndata_per_cpu)
            if ndata_cpu > 0
        ]

        with get_mp_executor(backend=mp_backend, n_workers=n_workers) as executor:
            futures = [executor.submit(_load_data, job) for job in jobs]

            # Propagate the first exception (if any)
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc is not None:
                    raise exc
        result = shared_arr

    return result

def read_sink(
        path: str, 
        iout: int,
        region: Region | np.ndarray | list | None = None, 
        icpu: int | None = None,
        target_fields: Sequence[str] | None = None,
        dtype: np.dtype | list | None = None,
        info: dict | None = None,
        exact_cut: bool=True) -> np.ndarray:

    if isinstance(region, np.ndarray) or isinstance(region, list):
        region = Box(region)

    if info is None:
        info = get_info(path, iout)

    output_dir = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout))
    if dtype is None:
        try:
            dtype = read_type_descriptor(path, iout, 'sink')
        except FileNotFoundError as e:
            fd_path = os.path.join(output_dir, config['FILE_DESCRIPTOR_FORMAT'].format(data='sink'))
            raise FileNotFoundError(
                f"File descriptor not found: {fd_path}\n"
                f"`dtype` may need to be provided manually.")

    dtype = np.dtype(dtype)
    dtype_out = dtype

    if target_fields is not None:
        dtype_out = np.dtype([(name, dtype_out.fields[name][0]) for name in target_fields if name in dtype_out.names])
    
    if icpu is None:
        sink_files = glob.glob(os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout), config['FILENAME_FORMAT_ANY'].format(data='sink', iout=iout)))
        if len(sink_files) == 0:
            return np.empty(0, dtype=dtype_out)
        filename = sink_files[0]
    else:
        filename = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout), config['FILENAME_FORMAT'].format(data='sink', iout=iout, icpu=icpu))
        if not os.path.exists(filename):
            return np.empty(0, dtype=dtype_out)

    with FortranFile(filename, mode="r") as f:
        nsink = f.read_ints('i4')
        result = np.empty(nsink, dtype=dtype_out)
        if nsink == 0:
            return result
        f.skip_records(1)

        for name in dtype.names:
            if name not in dtype_out.names:
                f.skip_records(1)
                continue

            dtype_format = dtype_out.fields[name][0]
            if np.issubdtype(dtype_format, np.integer):
                arr = f.read_ints(dtype_format)
            elif np.issubdtype(dtype_format, np.floating):
                arr = f.read_reals(dtype_format)
            else:
                raise TypeError(f"Unsupported data type for field '{name}': {dtype_format}")

            if arr.size != nsink:
                raise RuntimeError(
                    f"Unexpected size for field '{name}' in sink file: "
                    f"got {arr.size}, expected {nsink}"
                )
            result[name] = arr
    
    if exact_cut and region is not None:
        result = result[region.contains_data(result, cell=False)]

    return result
