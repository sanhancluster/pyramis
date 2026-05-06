import os
import glob
import h5py
import numpy as np
from typing import Callable, Sequence, overload, Literal, Tuple
import warnings

from concurrent.futures import as_completed
import configparser

import re
from .config_module import get_config, get_vname
from . import cgs_unit, timer, format_bytes, ANY, get_position_keys
from .astro import get_cosmo_table, cosmo_convert
from .core import compute_chunk_list_from_hilbert, str_to_tuple, quad_to_int
from .utils.hilbert import hilbert_to_compound, HILBERT_KEY_DTYPE
from pyramis.geometry import Region, Box
from .utils.fortranfile import FortranFile
from .utils.arrayview import ArrayView
from .utils import get_mp_executor

import numpy as np
import h5py
from multiprocessing.shared_memory import SharedMemory
from itertools import repeat

def _scheduled_snapshots(tout, time, t_thr, iout=None, report_missing=False):
    tout = np.unique(tout)
    tout = np.sort(tout)
    scheduled = np.zeros(len(time), dtype=bool)
    n_below, n_above, n_between = 0, 0, 0
    for t in tout:
        diff = np.abs(time - t)
        diff_masked = np.where(scheduled, np.inf, diff)
        cand_key = np.argmin(diff_masked)
        if np.abs(time[cand_key] - t) < t_thr[cand_key]:
            scheduled[cand_key] = True
        elif t < np.min(time):
            n_below += 1
        elif t > np.max(time):
            n_above += 1
        else:
            n_between += 1
            if report_missing:
                offset = (time[cand_key] - t)
                message = f"No snapshot found at {t:.5f} (closest is {time[cand_key]:.5f} with offset {offset:.5f}, threshold is {t_thr[cand_key]:.5f})"
                if iout is not None:
                    message = message[:-1] + f", at iout={iout[cand_key]})"
                timer.message(message)
    if n_below > 0 or n_above > 0 or n_between > 0:
        if report_missing:
            message = f"There are {n_below} missing snapshots below the first snapshot, {n_between} in between, {n_above} above the last snapshot."
            if iout is not None:
                message = message[:-1] + f", with iout range [{iout[np.argmin(time)]}, {iout[np.argmax(time)]}])"
            timer.message(message)
    return scheduled


def check_snapshots(path: str, check_data=['amr', 'hydro', 'part'], iout_min=None, iout_max=None, report_missing=False, namelist_path=None, scale_threshold=50.) -> np.ndarray:
    config = get_config()

    timer.start(f'Checking snapshots in {path} for {check_data}...')
    pattern = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=ANY))
    dirs = glob.glob(pattern)
    iout_list = []
    aexp_list = []
    time_list = []
    nstep_coarse_list = []
    aout, tout = None, None
    if 'cell' in check_data:
        check_data.remove('cell')
        check_data.extend(['amr', 'hydro'])
    for d in dirs:
        basename = os.path.basename(d)
        ok = True

        iout = int(basename.split('_')[-1])
        if iout_min is not None and iout < iout_min:
            continue
        if iout_max is not None and iout > iout_max:
            continue
        info_path = os.path.join(d, f'info_{iout:05d}.txt')

        if os.path.exists(info_path) is False:
            if report_missing:
                timer.message(f"Info file missing for iout={iout} in directory {d}.")
            continue
            
        info = parse_info(info_path)

        for data in check_data:
            file_pattern = os.path.join(d, config['FILENAME_FORMAT_RAMSES'].format(data=data, iout=iout, icpu=ANY))
            files = glob.glob(file_pattern)
            if len(files) != info['ncpu']:
                if report_missing:
                    timer.message(f"Number of '{data}' files does not match for iout={iout}. Expected {info['ncpu']} files, found {len(files)}.")
                ok = False                    
        if ok:
            iout_list.append(iout)
            aexp_list.append(info['aexp'])
            time_list.append(info['time'])
            nstep_coarse_list.append(info['nstep_coarse'])
            aout = info.get('aout', aout)
            tout = info.get('tout', tout)

    table = np.rec.fromarrays([iout_list, aexp_list, time_list, nstep_coarse_list, np.zeros(len(iout_list), dtype=bool)], dtype=[('iout', 'i4'), ('aexp', 'f8'), ('time', 'f8'), ('nstep_coarse', 'i4'), ('scheduled', '?')])
    table.sort(order='iout')
    
    if len(iout_list) == 0:
        return table

    if aout is None and tout is None:
        if namelist_path is None:
            namelist_path = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout_list[-1]), config['NAMELIST_FILENAME'])
        nml = parse_namelist(namelist_path)

        scheduled = np.zeros(len(table), dtype=bool)

        if 'aout' in nml['OUTPUT_PARAMS']:
            aout = np.array(tuple(nml['OUTPUT_PARAMS']['aout'].split(','))).astype(np.float64)
        if 'tout' in nml['OUTPUT_PARAMS']:
            tout = np.array(tuple(nml['OUTPUT_PARAMS']['tout'].split(','))).astype(np.float64)
    else:
        aout = np.array(aout)
        tout = np.array(tout)
    
    scheduled = np.zeros(len(table), dtype=bool)
    scheduled[table['iout'] == 1] = True # always include the first snapshot

    if aout is not None and len(aout) > 0 and not np.all(aout == 0.0):
        a_thr = table['aexp'] / table['nstep_coarse'] * scale_threshold
        scheduled |= _scheduled_snapshots(aout, table['aexp'], a_thr, iout=table['iout'], report_missing=report_missing)
    if tout is not None and len(tout) > 0 and not np.all(tout == 0.0):
        t_thr = table['time'] / table['nstep_coarse'] * scale_threshold
        scheduled |= _scheduled_snapshots(tout, table['time'], t_thr, iout=table['iout'], report_missing=report_missing)

    table['scheduled'] = scheduled
    timer.record(f'Checked snapshots in {path} for {check_data}. Found {len(table)} snapshots, with {np.sum(scheduled)} scheduled in namelist.')

    return table


def read_type_descriptor(path: str, iout: int, data: str='part') -> np.dtype:
    config = get_config()
    timer.message(f"Reading type descriptor for {data} at iout={iout} from {path}...", 2)
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
    timer.message(f"Parsing info file: {path}...", 2)
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


def read_info(output_path, iout: int | None=None, namelist_path=None, cosmo=True, cosmo_table=None, read_amr=True, read_hydro=True) -> dict:
    config = get_config()
    if iout is None:
        files = glob.glob(os.path.join(output_path, config['OUTPUT_FORMAT'].format(iout=ANY), f'info_{ANY:05d}.txt'))
        if len(files) == 0:
            raise FileNotFoundError(f"No info file found in {output_path}.")
        info_path = files[0]
    else:
        info_path = os.path.join(output_path, config['OUTPUT_FORMAT'].format(iout=iout), f'info_{iout:05d}.txt')
    timer.message(f"Getting info for iout={iout} from {output_path}...", 2)
    info = parse_info(info_path)
    info['iout'] = iout

    if read_amr:
        amr_files = glob.glob(os.path.join(output_path, config['OUTPUT_FORMAT'].format(iout=iout), config['FILENAME_FORMAT_RAMSES'].format(data='amr', iout=iout, icpu=ANY)))
        if len(amr_files) == 0:
            raise FileNotFoundError(f"No AMR file found at iout = {iout} in {output_path}.")

        amr_path = amr_files[0]
        timer.message(f"Reading AMR file: {amr_path}...", 2)
        with FortranFile(amr_path, mode='r') as f:
            info['ncpu'], = f.read_ints()
            info['ndim'], = f.read_ints()
            info['nx'], info['ny'], info['nz'], = f.read_ints()
            info['nlevelmax'], = f.read_ints()
            info['ngridmax'], = f.read_ints()
            info['nboundary'], = f.read_ints()
            info['ngrid_current'] = f.read_ints()
            info['boxlen'], = f.read_reals()

            info['noutput'], info['idout'], info['ifout'], = f.read_ints()
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
                    # quad case: 16-byte IEEE 754 quad float → compound
                    info['bounds'] = hilbert_to_compound(quad_to_int(bounds))
                else:
                    # double case: float64 values are exact integers at these magnitudes
                    info['bounds'] = hilbert_to_compound(
                        np.array([int(x) for x in bounds.view('f8')], dtype=object)
                    )

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
        hydro_files = glob.glob(os.path.join(output_path, config['OUTPUT_FORMAT'].format(iout=iout), config['FILENAME_FORMAT_RAMSES'].format(data='hydro', iout=iout, icpu=ANY)))
        if len(hydro_files) > 0:
            hydro_path = hydro_files[0]
            timer.message(f"Reading hydro file: {hydro_path}...", 2)
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
        info['age'] = cosmo_convert(info['cosmo_table'], info['aexp'], 'aexp', 'age') / cgs_unit['Gyr']['factor']
        info['lookback_time'] = cosmo_convert(info['cosmo_table'], 1.0, 'aexp', 'age') / cgs_unit['Gyr']['factor'] - info['age']
        info['z'] = 1.0 / info['aexp'] - 1.0

    return info


def get_data_path(data_name, path, iout, icpu):
    config = get_config()
    return os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout), config['FILENAME_FORMAT_RAMSES'].format(data=data_name, iout=iout, icpu=icpu))

def _read_npart_file(path, iout, icpu, part_type, family_exists):
    filename = get_data_path('part', path, iout, icpu)
    with FortranFile(f"{filename}", mode='r') as f:
        f.skip_records(2)
        npart, = f.read_ints('i4')
        
        # Option 1: if part_type is None, just sum npart
        if part_type is None:
            return npart
        f.skip_records(5)

        # Option 2: read family/epoch and classify
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
    config = get_config()
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


def read_npart_per_cpu(path, iout, cpulist=None, dtype_read=None, part_type=None, info=None, n_workers: int | None=None, mp_backend: str="thread") -> np.ndarray:
    timer.message(f"Reading number of particles per CPU for iout={iout} from {path}...", 2)
    if cpulist is None:
        if info is None:
            info = read_info(path, iout)
        cpulist = np.arange(1, int(info['ncpu'])+1)
    if dtype_read is None:
        dtype_read = read_type_descriptor(path, iout, 'part')
    family_exists = dtype_read.names is not None and 'family' in dtype_read.names

    npart_cpu = []
    if n_workers in (None, 1):
        for icpu in cpulist:
            npart_cpu.append(_read_npart_file(path, iout, icpu, part_type, family_exists))
        return np.array(npart_cpu)
    
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
                )
            )
        return np.array(results)


def mask_by_part_type(part, part_type):
    config = get_config()
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


def read_part(
        path: str, 
        iout: int | None = None, 
        region: Region | np.ndarray | list | None = None, 
        cpulist: Sequence[int] | np.ndarray | None = None,
        target_fields: Sequence[str] | None = None,
        part_type: str | None=None, 
        dtype: np.dtype | list | None = None,
        info: dict | None = None,
        read_cpu=False,
        exact_cut: bool=True,
        n_workers: int | None=None,
        use_process: bool=False,
        return_view: bool=True) -> ArrayView:

    config = get_config()
    timer.start(f"Reading particle data from {path} at iout={iout}...")

    if n_workers is None:
        n_workers = config['DEFAULT_N_PROCS']

    if isinstance(region, np.ndarray) or isinstance(region, list):
        region = Box(region)

    if info is None:
        info = read_info(path, iout)
    
    if use_process:
        mp_backend = "process"
    else:
        mp_backend = "thread"

    if iout is not None:
        output_dir = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout))
    else:
        output_dir = path

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
        if 'family' not in dtype_read.names:
            if 'tform' not in target_fields and 'tform' in dtype_read.names:
                warnings.warn("Including 'tform' field for classification.", UserWarning)
                target_fields = target_fields + ['tform']
            if 'm' not in target_fields and 'm' in dtype_read.names:
                warnings.warn("Including 'm' field for classification.", UserWarning)
                target_fields = target_fields + ['m']
            if 'id' not in target_fields and 'id' in dtype_read.names:
                warnings.warn("Including 'id' field for classification.", UserWarning)
                target_fields = target_fields + ['id']
        elif 'family' not in target_fields:
            # Ensure 'family' is included for classification
            warnings.warn("Including 'family' field for classification.", UserWarning)
            target_fields = target_fields + ['family']

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
            n_workers=n_workers
        ) + 1
    elif cpulist is None:
        cpulist = np.arange(1, int(info['ncpu'])+1)
    else:
        cpulist = np.array(cpulist)

    npart_per_cpu = read_npart_per_cpu(path, iout, cpulist, dtype_read=dtype_read, part_type=part_type)
    npart = np.sum(npart_per_cpu) if len(npart_per_cpu) > 0 else 0
    size_byte = npart * dtype_out.itemsize
    timer.message(f"Total number of particles to read: {npart} ({format_bytes(size_byte)}) across {len(cpulist)} / {int(info['ncpu'])} files.")
    if npart == 0:
        return np.empty(0, dtype=dtype_out)
    
    args = path, iout, dtype_read, part_type
    
    if n_workers == 1:
        result = _read_from_cpulist(
            args,
            cpulist,
            dtype_out,
            npart_per_cpu,
            _load_part_file
        )
    else:
        result = _read_from_cpulist_mp(
            args,
            cpulist,
            dtype_out,
            npart_per_cpu,
            _load_part_file,
            n_workers=n_workers,
            mp_backend=mp_backend,
            return_view=return_view
        )
    
    if exact_cut and region is not None:
        result2 = result[region.contains_data(result, cell=False)]
        if isinstance(result, ArrayView):
            result.close()
        result = result2
    timer.record(f"Finished reading particle data from {path} at iout={iout}. Found {len(result)} particles.")
    if isinstance(result, ArrayView):
        result.info = info
    elif return_view:
        result = ArrayView(result, info=info)
    return result


def _read_with_format(f, data, dtype_read):
    dtype_out = data.dtype
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
            raise TypeError(f"Unsupported data type: {dtype_format}")

        if arr.size != data.shape[0]:
            raise RuntimeError(
                f"Unexpected size for field '{name}' on CPU {icpu}: "
                f"got {arr.size}, expected {data.shape[0]}"
            )
        data[name][:] = arr


def _load_part_file(icpu, output_arr, path, iout, dtype_read, part_type=None):
    config = get_config()
    dtype_out = output_arr.dtype
    filename = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout), config['FILENAME_FORMAT_RAMSES'].format(data='part', iout=iout, icpu=icpu))

    with FortranFile(filename, mode="r") as f:
        f.skip_records(2)
        npart_file, = f.read_ints('i4')
        f.skip_records(5)

        if part_type is None:
            part_data = output_arr
        else:
            # npart_cpu may be larger than npart
            part_data = np.empty(npart_file, dtype=dtype_out)

        _read_with_format(f, part_data, dtype_read)
        
        if get_vname('cpu') in dtype_out.names:
            part_data[get_vname('cpu')] = icpu

        if part_type is not None:
            mask = mask_by_part_type(part_data, part_type)
            part_data = part_data[mask]
            output_arr[:] = part_data


def read_ncell_per_cpu(path, iout, cpulist=None, info=None, read_branch=False, levelmax=None) -> np.ndarray:
    config = get_config()
    timer.message(f"Reading number of cells per CPU for iout={iout} from {path}...", 2)
    if info is None:
        info = read_info(path, iout)
    if cpulist is None:
        info = read_info(path, iout)
        cpulist = np.arange(1, int(info['ncpu'])+1)

    ndim = info['ndim']
    ncpu = info['ncpu']
    nlevelmax = info['nlevelmax']
    nlevelmax_read = nlevelmax if levelmax is None else min(nlevelmax, levelmax)
    nboundary = info['nboundary']
    twotondim = 2 ** ndim

    ncell_cpu = []
    for icpu in cpulist:
        filename = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout), config['FILENAME_FORMAT_RAMSES'].format(data='amr', iout=iout, icpu=icpu))
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
            levels, cpus = np.nonzero(ngridfile)
            for ilevel, jcpu in zip(levels + 1, cpus + 1):
                f.skip_records(3)
                if jcpu == icpu:
                    if ilevel == nlevelmax_read:
                        ncell +=  ngridfile[ilevel-1, jcpu-1] * twotondim
                        break
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
    return np.array(ncell_cpu)


def read_cell(
        path: str, 
        iout: int | None = None, 
        region: Region | np.ndarray | list | None = None, 
        cpulist: Sequence[int] | np.ndarray | None = None,
        target_fields: Sequence[str] | None = None,
        levelmax = None,
        dtype_hydro = None,
        info: dict | None = None,
        read_hydro=True,
        read_grav=False,
        read_cpu=False,
        read_branch=False,
        exact_cut: bool=True,
        n_workers: int | None = None,
        use_process: bool=False,
        return_view: bool=True) -> ArrayView:

    config = get_config()
    timer.start(f"Reading cell data from {path} at iout={iout}...")

    if isinstance(region, np.ndarray) or isinstance(region, list):
        region = Box(region)

    if info is None:
        info = read_info(path, iout)

    if n_workers is None:
        n_workers = config['DEFAULT_N_PROCS']

    if use_process:
        mp_backend = "process"
    else:
        mp_backend = "thread"

    if iout is not None:
        output_name = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout))
    else:
        output_name = path

    pos_dtype = [(key, np.float64) for key in get_position_keys()[:info['ndim']]]
    descr_out = pos_dtype + [(get_vname('level'), np.int32)]

    if read_hydro:
        fd_path = os.path.join(output_name, config['FILE_DESCRIPTOR_FORMAT'].format(data='hydro'))
        if dtype_hydro is None:
            try:
                dtype_hydro = read_type_descriptor(path, iout, 'hydro')
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"File descriptor not found: {fd_path}\n"
                    f"`dtype` may need to be provided manually. (e.g., [('rho', 'f8'), ('vx', 'f8'), ...])")

        dtype_hydro = np.dtype(dtype_hydro)
        descr_out = descr_out + dtype_hydro.descr
    
    if read_grav:
        descr_out = descr_out + [(get_vname('potential'), np.float64)]
    
    if read_cpu:
        descr_out = descr_out + [(get_vname('cpu'), np.int32)]
    
    dtype_out = np.dtype(descr_out)

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
            n_workers=n_workers
        ) + 1
    elif cpulist is None:
        cpulist = np.arange(1, int(info['ncpu'])+1)
    else:
        cpulist = np.array(cpulist)
    
    ncell_per_cpu = read_ncell_per_cpu(path, iout, cpulist, info=info, read_branch=read_branch, levelmax=levelmax)
    ncell = np.sum(ncell_per_cpu) if len(ncell_per_cpu) > 0 else 0
    size_byte = ncell * dtype_out.itemsize
    timer.message(f"Total number of cells to read: {ncell} ({format_bytes(size_byte)}) across {len(cpulist)} / {int(info['ncpu'])} files.")
    if ncell == 0:
        return ArrayView(np.empty(0, dtype=dtype_out), info=info)
    
    args = (path, iout, dtype_hydro, read_hydro, read_grav, read_branch, info, levelmax)

    if n_workers == 1:
        result = _read_from_cpulist(
            args,
            cpulist,
            dtype_out,
            ncell_per_cpu,
            _load_cell_file
        )
    
    else:
        result = _read_from_cpulist_mp(
            args,
            cpulist,
            dtype_out,
            ncell_per_cpu,
            _load_cell_file,
            n_workers=n_workers,
            mp_backend=mp_backend,
            return_view=return_view
        )
    
    if exact_cut and region is not None:
        result2 = result[region.contains_data(result, cell=True, boxlen=info['boxlen'])]
        if isinstance(result, ArrayView):
            result.close()
        result = result2
    timer.record(f"Finished reading cell data from {path} at iout={iout}. Found {len(result)} cells.")
    if isinstance(result, ArrayView):
        result.info = info
    elif return_view:
        result = ArrayView(result, info=info)

    return result

def _load_cell_file(icpu, output_arr, path, iout, dtype_hydro, read_hydro=True, read_grav=False, read_branch=False, info=None, levelmax=None):
    config = get_config()
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
        info = read_info(path, iout)
    
    dtype_out = output_arr.dtype
    
    ndim = info['ndim']
    ncpu = info['ncpu']
    nlevelmax = info['nlevelmax']
    nlevelmax_read = nlevelmax if levelmax is None else min(levelmax, nlevelmax)
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

    filename_amr = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout), config['FILENAME_FORMAT_RAMSES'].format(data='amr', iout=iout, icpu=icpu))
    filename_hydro = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout), config['FILENAME_FORMAT_RAMSES'].format(data='hydro', iout=iout, icpu=icpu))
    filename_grav = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout), config['FILENAME_FORMAT_RAMSES'].format(data='grav', iout=iout, icpu=icpu))

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

        for ilevel in range(1, nlevelmax_read + 1):
            nloop_before = np.count_nonzero(ngridfile[ilevel - 1, :icpu - 1])
            nloop_after = np.count_nonzero(ngridfile[ilevel - 1, icpu:])
            ncache = ngridfile[ilevel - 1, icpu - 1]
            f_amr.skip_records((3 + skip_amr) * nloop_before)
            if ncache > 0:
                f_amr.skip_records(3)

                pos = [] # list of position arrays
                pos_keys = get_position_keys()
                for idim in range(ndim):
                    if pos_keys[idim] in dtype_out.names:
                        p = f_amr.read_reals(np.float64)
                        pos.append((p + oct_offset_local[:, idim] / 2**ilevel - coarse_min[idim]) * boxlen)
                    else:
                        f_amr.skip_records(1)
                        pos.append(None)
                
                f_amr.skip_records(2 * ndim + 1)
                son = f_amr.read_arrays(twotondim)
                f_amr.skip_records(2 * twotondim)

                if ilevel == nlevelmax_read:
                    ok = np.ones_like(son, dtype=bool)
                else:
                    ok = (son == 0) if not read_branch else (son != 0)

                # save the indices of the cells to be read
                iread, jread = np.nonzero(ok)
                nread = iread.size
                jread_arr[cursor:cursor + nread] = jread

                # count number of cells read at this level and octant
                nread_arr[ilevel - 1] = np.sum(ok, axis=1)

                for idim in range(ndim):
                    key = pos_keys[idim]
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
            for ilevel in range(1, nlevelmax_read + 1):
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

                for ilevel in range(1, nlevelmax_read + 1):
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


def _read_from_cpulist(
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

        # call the loading function for this CPU slice
        func(
            icpu,
            data[offset:offset + ndata_cpu],
            *args
        )
    return data


def _load_data_with_func(
        ifile: int,
        shm_name: str | None,
        shared_arr: np.ndarray | None,
        dtype_out: np.dtype,
        ndata_tot: int,
        offset: int,
        ndata_cpu: int,
        func: Callable,
        *func_args) -> int:

    if ndata_cpu == 0:
        return 0

    # Attach shared memory
    if shm_name is not None:
        shm = SharedMemory(name=shm_name)
        try:
            shared_arr = np.ndarray(
                (ndata_tot,),
                dtype=dtype_out,
                buffer=shm.buf,
            )
            # view for this CPU slice
            output_arr = shared_arr[offset:offset + ndata_cpu]
            func(ifile, output_arr, *func_args)
        finally:
            shm.close()
    elif shared_arr is not None:
        output_arr = shared_arr[offset:offset + ndata_cpu]
        func(ifile, output_arr, *func_args)

    return ndata_cpu


def _read_from_cpulist_mp(
        func_args: Tuple,
        cpulist: Sequence[int] | np.ndarray,
        dtype_out: np.dtype,
        ndata_per_cpu: Sequence[int],
        func: Callable,
        n_workers: int,
        mp_backend: str,
        return_view: bool) -> np.ndarray | ArrayView:
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
                (int(icpu), shm.name, None, dtype_out, ndata, int(offset), int(ndata_cpu), func, *func_args)
                for icpu, offset, ndata_cpu in zip(cpulist, offsets, ndata_per_cpu)
                if ndata_cpu > 0]

            with get_mp_executor(backend=mp_backend, n_workers=n_workers) as executor:
                futures = [executor.submit(_load_data_with_func, *job) for job in jobs]

                # Propagate the first exception (if any)
                for fut in as_completed(futures):
                    exc = fut.exception()
                    if exc is not None:
                        raise exc

            # At this point, shared_arr is fully populated with all data
            if return_view:
                # Return the shared view; caller must manage shm lifetime
                result = ArrayView(shm, (ndata,), dtype_out)
            else:
                result = np.array(shared_arr, copy=True)
        finally:
            # Clean up shared memory if we own it (copy_result=True).
            if not return_view:
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
            (int(icpu), None, shared_arr, dtype_out, ndata, int(offset), int(ndata_cpu), func, *func_args)
            for icpu, offset, ndata_cpu in zip(cpulist, offsets, ndata_per_cpu)
            if ndata_cpu > 0
        ]

        with get_mp_executor(backend=mp_backend, n_workers=n_workers) as executor:
            futures = [executor.submit(_load_data_with_func, *job) for job in jobs]

            # Propagate the first exception (if any)
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc is not None:
                    raise exc
        result = shared_arr

    return result


# Functions for reading sink files
def read_sink(
        path: str, 
        iout: int | None = None,
        region: Region | np.ndarray | list | None = None, 
        icpu: int | None = None,
        target_fields: Sequence[str] | None = None,
        dtype: np.dtype | list | None = None,
        info: dict | None = None,
        exact_cut: bool = True,
        return_view: bool = True) -> ArrayView:

    config = get_config()
    timer.start(f"Reading sink data from {path} at iout={iout}...")

    if isinstance(region, np.ndarray) or isinstance(region, list):
        region = Box(region)

    if info is None:
        info = read_info(path, iout)

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
        sink_files = glob.glob(os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout), config['FILENAME_FORMAT_RAMSES'].format(data='sink', iout=iout, icpu=ANY)))
        if len(sink_files) == 0:
            return np.empty(0, dtype=dtype_out)
        filename = sink_files[0]
    else:
        filename = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout), config['FILENAME_FORMAT_RAMSES'].format(data='sink', iout=iout, icpu=icpu))
        if not os.path.exists(filename):
            return np.empty(0, dtype=dtype_out)

    with FortranFile(filename, mode="r") as f:
        nsink = f.read_ints('i4')
        result = np.empty(nsink, dtype=dtype_out)
        if nsink == 0:
            if info is not None and return_view:
                result = ArrayView(result, info=info)
            return result
        f.skip_records(1)

        _read_with_format(f, result, dtype)
    
    if exact_cut and region is not None:
        result = result[region.contains_data(result, cell=False)]

    timer.record(f"Finished reading sink data from {filename}. Found {len(result)} sink particles.")
    if return_view:
        result = ArrayView(result, info=info)
    return result


# Functions for reading sink properties
def _load_sinkprops_file(icoarse, output_arr, path, dtype_read):
    config = get_config()
    filename = os.path.join(path, config['FILENAME_FORMAT_SINKPROPS'].format(icoarse=icoarse))
    with FortranFile(filename, mode='r') as f:
        f.skip_records(2)
        aexp = f.read_reals()
        unit_l = f.read_reals()
        unit_d = f.read_reals()
        unit_t = f.read_reals()
        output_arr[get_vname('icoarse')][:] = icoarse
        output_arr[get_vname('aexp')][:] = aexp
        output_arr[get_vname('unit_l')][:] = unit_l
        output_arr[get_vname('unit_d')][:] = unit_d
        output_arr[get_vname('unit_t')][:] = unit_t
        _read_with_format(f, output_arr[:], dtype_read)


def read_nsinkprops_per_file(path: str, icoarse_read: Sequence[int] | np.ndarray) -> np.ndarray:
    config = get_config()
    nsink_per_file = []
    for icoarse in icoarse_read:
        filename = os.path.join(path, config['FILENAME_FORMAT_SINKPROPS'].format(icoarse=icoarse))
        with FortranFile(filename, mode='r') as f:
            nsink = f.read_ints('i4')[0]
            nsink_per_file.append(nsink)
    return np.array(nsink_per_file)


def read_sinkprops(
        path: str,
        icoarse_min: int | None = None,
        icoarse_max: int | None = None,
        dtype: np.dtype | list | None = None,
        n_workers: int | None = None,
        use_process: bool = True,
        return_view: bool = True) -> ArrayView | np.ndarray:

    config = get_config()    
    timer.record(f"Reading sink properties from {path}...")
    try:
        info = read_info(path, iout=None)
    except FileNotFoundError:
        info = None

    if n_workers is None:
        n_workers = config['DEFAULT_N_PROCS']
    
    if use_process:
        mp_backend = "process"
    else:
        mp_backend = "thread"

    sinkprops_avail = glob.glob(os.path.join(path, config['FILENAME_FORMAT_SINKPROPS'].format(icoarse=ANY)))
    icoarse_avail = np.array([int(os.path.basename(f).split('_')[1].split('.')[0]) for f in sinkprops_avail])
    if icoarse_max is not None and icoarse_max < 0:
        icoarse_max = np.max(icoarse_avail) + icoarse_max + 1
    if icoarse_min is not None and icoarse_min < 0:
        icoarse_min = np.max(icoarse_avail) + icoarse_min + 1
    icoarse_read = icoarse_avail[
        (icoarse_avail >= (icoarse_min if icoarse_min is not None else -np.inf)) &
        (icoarse_avail <= (icoarse_max if icoarse_max is not None else np.inf))
    ]

    # dtype for formatted reading from sinkprops files; must be consistent with the file format
    if dtype is None:
        dtype = [(get_vname(field[0]), field[1]) for field in config['SINKPROPS_DTYPE']]
    dtype = np.dtype(dtype)

    # output dtype
    dtype_out = np.dtype([(get_vname('icoarse'), np.int32), (get_vname('aexp'), np.float64), (get_vname('unit_l'), np.float64), (get_vname('unit_d'), np.float64), (get_vname('unit_t'), np.float64)] + dtype.descr)

    size = np.sum([os.path.getsize(os.path.join(path, config['FILENAME_FORMAT_SINKPROPS'].format(icoarse=icoarse))) for icoarse in icoarse_read])
    timer.message(f"Found {len(icoarse_read)} sink property files to read ({format_bytes(size)}).")

    if len(icoarse_read) == 0:
        out = np.empty(0, dtype=dtype_out)
        if info is not None and return_view:        
            return ArrayView(out, info=info)
        else:
            return out
    
    # get number of sink particles per file
    nsink_per_file = read_nsinkprops_per_file(path, icoarse_read)
    
    # Total number of sink particles across all files
    ndata_tot = np.sum(nsink_per_file)
    if ndata_tot == 0:
        out = np.empty(0, dtype=dtype_out)
        if info is not None and return_view:
            return ArrayView(out, info=info)
        else:
            return out
    
    # Precompute offsets for each file
    offsets = np.zeros_like(nsink_per_file)
    offsets[1:] = np.cumsum(nsink_per_file[:-1])
    offsets = offsets.astype(int)

    itemsize = dtype_out.itemsize
    total_bytes = ndata_tot * itemsize
    func_args = path, dtype

    if mp_backend == "process" and n_workers > 1:
        shm = SharedMemory(create=True, size=total_bytes)
        try:
            shared_arr = np.ndarray((ndata_tot,), dtype=dtype_out, buffer=shm.buf)

            # Build job list for each file
            jobs = [
                (int(icoarse), shm.name, None, dtype_out, ndata_tot, int(offset), int(nsink_per_file[idx]), _load_sinkprops_file, *func_args)
                for idx, (icoarse, offset) in enumerate(zip(icoarse_read, offsets))
            ]

            with get_mp_executor(backend="process", n_workers=n_workers) as executor:
                futures = [executor.submit(_load_data_with_func, *job) for job in jobs]

                # Propagate the first exception (if any)
                for fut in as_completed(futures):
                    exc = fut.exception()
                    if exc is not None:
                        raise exc

            result = np.array(shared_arr, copy=True)
            if return_view:
                result = ArrayView(shm, (ndata_tot,), dtype_out)
            else:
                result = np.array(shared_arr, copy=True)

        finally:
            if not return_view:
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
            (int(icoarse), None, shared_arr, dtype_out, ndata_tot, int(offset), int(nsink_per_file[idx]), _load_sinkprops_file, *func_args)
            for idx, (icoarse, offset) in enumerate(zip(icoarse_read, offsets))
        ]
        with get_mp_executor(backend="thread", n_workers=n_workers) as executor:
            futures = [executor.submit(_load_data_with_func, *job) for job in jobs]
            # Propagate the first exception (if any)
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc is not None:
                    raise exc
        result = shared_arr
    timer.record(f"Finished reading sink properties from {path}. Found {len(result)} items.")
    
    if info is not None and return_view:
        return ArrayView(result, info=info)
    else:
        return result

        