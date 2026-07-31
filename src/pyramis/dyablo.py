from fileinput import filename
import warnings

from .utils.arrayview import ArrayView

from . import config, timer, ANY, get_position_names
from .geometry import Box
import h5py
import os
import numpy as np
import glob
import configparser

def check_snapshots(path, prefix=None):
    if prefix is None:
        prefix = ANY
    timer.start(f"Checking for Dyablo snapshots in {path} with prefix {prefix}...")
    pattern = os.path.join(path, config['FILENAME_FORMAT_DYABLO'].format(prefix=prefix, istep=ANY))
    files = glob.glob(pattern)
    scalar_data_dtypes = {}
    
    for file in files:
        with h5py.File(file, 'r') as f:
            if 'scalar_data' in f.keys():
                scalar_data = f['scalar_data'].attrs
                scalar_data_dtypes.update({name: scalar_data[name].dtype for name in scalar_data.keys()})
            else:
                warnings.warn(f"No scalar_data found in file {file}.")
                scalar_data_dtypes = {}
 
    dtype = [('istep', 'i4')]
    for name, dtype_value in scalar_data_dtypes.items():
        dtype.append((name, dtype_value))
    snapshots = np.zeros(len(files), dtype=dtype)

    for i, file in enumerate(files):
        filename = os.path.basename(file)
        try:
            istep_str = filename.split('_iter')[1].split('.h5')[0]
            istep = int(istep_str)
        except (IndexError, ValueError):
            warnings.warn(f"Could not extract istep from filename {filename}. Skipping.")
            continue

        with h5py.File(file, 'r') as f:
            if 'scalar_data' in f.keys():
                scalar_data = f['scalar_data'].attrs
                snapshots['istep'][i] = istep
                for name in scalar_data.keys():
                    snapshots[name][i] = scalar_data[name]
    timer.record(f"Found {len(snapshots)} Dyablo snapshots in {path} with prefix {prefix}.")
    return snapshots


def read_info(path, iout=None, prefix=None):
    if prefix is None:
        prefix = ANY
    timer.start(f"Reading Dyablo info from {path} with prefix {prefix}...")
    pattern = os.path.join(path, config['FILENAME_FORMAT_DYABLO'].format(prefix=prefix, istep=ANY))
    files = glob.glob(pattern)
    if len(files) == 0:
        raise FileNotFoundError(f"No files found matching pattern {pattern}.")
    files.sort()
    if iout is None:
        iout = -1
    if iout > 0:
        filename = files[iout-1]
    else:
        filename = files[iout]

    with h5py.File(filename, 'r') as f:
        if 'scalar_data' in f.keys():
            info = dict(f['scalar_data'].attrs)
        else:
            info = {}

    timer.record(f"Finished reading Dyablo info from {filename}.")
    return info

def read_ini(path, filename_ini=None):
    if filename_ini is None:
        filename_ini = config['LAST_INI_FILENAME']
    
    path_dir = os.path.dirname(path) if os.path.isfile(path) else path
    path_ini = os.path.join(path_dir, filename_ini)
    print(path_ini)
    if os.path.exists(path_ini):
        cp = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))
        cp.read(path_ini)
        print(cp)
        return cp
    else:
        return None


def find_snapshot(path, iout=None, istep=None, prefix=None):
    if prefix is None:
        prefix = ANY
    if istep is None:
        istep = ANY
    if iout is None:
        iout = ANY

    if isinstance(iout, int) or isinstance(istep, int):
        pattern = os.path.join(path, config['FILENAME_FORMAT_DYABLO'].format(prefix=prefix, istep=istep))

        files = glob.glob(pattern)
        files.sort()
        if len(files) == 0:
            raise FileNotFoundError(f"No files found matching pattern {pattern}.")
        if isinstance(iout, int):
            if iout > 0:
                iout -= 1
            elif iout == 0:
                raise ValueError("iout cannot be 0. It should be a positive or negative integer.")
            filename = files[iout]
        elif len(files) > 1:
            raise FileExistsError(f"Multiple files found matching pattern {pattern}. Please specify a prefix.")
        else:
            filename = files[0]
    else:
        filename = path
    return filename


def read_cell(path, iout=None, istep=None, prefix=None, filename_ini=None, return_view=True):
    filename = find_snapshot(path, iout=iout, istep=istep, prefix=prefix)
    timer.start(f"Reading Dyablo cell data from {filename}...")

    cp = read_ini(path, filename_ini=filename_ini)
    ndim = int(cp['mesh']['ndim']) if cp is not None else 3
    if cp is not None:
        coarse_bins = int(cp['amr']['bx']), int(cp['amr']['by']), int(cp['amr']['bz'])
        coarse_bins = np.asarray(coarse_bins)
        domain_lims = np.asarray([[float(cp['mesh']['xmin']), float(cp['mesh']['xmax'])],
                                  [float(cp['mesh']['ymin']), float(cp['mesh']['ymax'])],
                                  [float(cp['mesh']['zmin']), float(cp['mesh']['zmax'])]])[:ndim]
        region = Box(box=[[float(cp['mesh']['xmin']), float(cp['mesh']['xmax'])],
                          [float(cp['mesh']['ymin']), float(cp['mesh']['ymax'])],
                          [float(cp['mesh']['zmin']), float(cp['mesh']['zmax'])]][:ndim], domain_lims=domain_lims)
        domain_extent = domain_lims[:, 1] - domain_lims[:, 0]
    else:
        coarse_bins = np.array(config['DEFAULT_COARSE_BINS_DYABLO'])
        region = Box()
        domain_lims = None
        domain_extent = None
        warnings.warn(f"file {filename_ini} not found in the same directory with the snapshot. Using default value of {coarse_bins} for bx, by, bz.")

    with h5py.File(filename, 'r') as f:
        connectivity = f['connectivity'][:]
        coordinates = f['coordinates'][:]
        n_vertices = connectivity.shape[1]
        vertices = np.reshape(coordinates[connectivity], (-1, n_vertices, 3))
        if domain_extent is None:
            domain_extent = np.max(coordinates, axis=0) - np.min(coordinates, axis=0)
        centers = np.mean(vertices, axis=1)
        dtypes = [('position_x', 'f8'), ('position_y', 'f8'), ('position_z', 'f8')]
        if 'level' not in f.keys():
            i_dim = 0
            vertices_min = np.min(vertices, axis=1)
            vertices_max = np.max(vertices, axis=1)
            cell_extents = (vertices_max - vertices_min)[:, i_dim] / domain_extent[i_dim]
            levels = np.round(-np.log2(cell_extents * coarse_bins[i_dim])).astype(int)
            dtypes += [('level', 'i4')]
        else:
            levels = None

        n_data = connectivity.shape[0]

        keys = [k for k in f.keys() if k not in ['connectivity', 'coordinates', 'scalar_data']]

        for key in keys:
            dtypes.append((key, f[key].dtype))
        dtypes = np.dtype(dtypes)
        table = np.empty(n_data, dtype=dtypes)

        for key in keys:
            table[key] = f[key][:]

        table['position_x'] = centers[:, 0]
        table['position_y'] = centers[:, 1]
        table['position_z'] = centers[:, 2]

        if levels is not None:
            if levels.ndim != table['level'].ndim:
                levels = levels[:, 0]
            table['level'] = levels

    timer.record(f"Finished reading Dyablo cell data from {filename}. Found {n_data} cells.")
    if return_view:
        info = dict(cp) if cp is not None else {}
        info.update(read_info(path, iout=iout, prefix=prefix))
        info['ndim'] = ndim
        info['coarse_bins'] = coarse_bins[:info['ndim']]
        return ArrayView(table, info=info, region=region)
    else:
        return table


def read_part(path, iout=None, istep=None, prefix=None, filename_ini=None, return_view=True):
    filename = find_snapshot(path, iout=iout, istep=istep, prefix=prefix)
    timer.start(f"Reading Dyablo particle data from {filename}...")

    cp = read_ini(path, filename_ini=filename_ini)
    ndim = int(cp['mesh']['ndim']) if cp is not None else 3
    if cp is not None:
        coarse_bins = int(cp['amr']['bx']), int(cp['amr']['by']), int(cp['amr']['bz'])
        coarse_bins = np.asarray(coarse_bins)
        domain_lims = np.asarray([[float(cp['mesh']['xmin']), float(cp['mesh']['xmax'])],
                                  [float(cp['mesh']['ymin']), float(cp['mesh']['ymax'])],
                                  [float(cp['mesh']['zmin']), float(cp['mesh']['zmax'])]])[:ndim]
        region = Box(box=[[float(cp['mesh']['xmin']), float(cp['mesh']['xmax'])],
                          [float(cp['mesh']['ymin']), float(cp['mesh']['ymax'])],
                          [float(cp['mesh']['zmin']), float(cp['mesh']['zmax'])]][:ndim], domain_lims=domain_lims)
    else:
        coarse_bins = np.array(config['DEFAULT_COARSE_BINS_DYABLO'])
        domain_lims = None
        region = Box()
        warnings.warn(f"file {filename_ini} not found in the same directory with the snapshot. Using default value of {coarse_bins} for bx, by, bz.")

    with h5py.File(filename, 'r') as f:
        # add coordinates if they exist
        if 'coordinates' in f.keys():
            centers = f['coordinates'][:]
            ndim = centers.shape[1]
            position_names = get_position_names(ndim=ndim)
            dtypes = [(name, 'f8') for name in position_names]
        else:
            centers = None
            dtypes = []


        keys = [k for k in f.keys() if k not in ['scalar_data', 'coordinates']]
        for key in keys:
            dtypes.append((key, f[key].dtype))
        dtypes = np.dtype(dtypes)
        n_data = f[keys[0]].shape[0]
        table = np.empty(n_data, dtype=dtypes)

        for key in keys:
            table[key] = f[key][:]

        if centers is not None:
            for i, name in enumerate(position_names):
                table[name] = centers[:, i]

    timer.record(f"Finished reading Dyablo particle data from {filename}. Found {n_data} particles.")
    if return_view:
        info = dict(cp) if cp is not None else {}
        info.update(read_info(path, iout=iout, prefix=prefix))
        info['ndim'] = ndim
        return ArrayView(table, info=info, region=region)
    else:
        return table