import warnings

from . import config, timer, ANY
import h5py
import os
import numpy as np
import glob

def check_snapshots(path, prefix=None):
    timer.start(f"Checking for Dyablo snapshots in {path} with prefix {prefix}...")
    if prefix is None:
        prefix = ANY
    pattern = os.path.join(path, config['FILENAME_FORMAT_DYABLO'].format(prefix=prefix, istep=ANY))
    files = glob.glob(pattern)
    scalar_data_dtypes = {}
    
    for file in files:
        with h5py.File(file, 'r') as f:
            scalar_data = f['scalar_data'].attrs
            scalar_data_dtypes.update({name: scalar_data[name].dtype for name in scalar_data.keys()})
 
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
            scalar_data = f['scalar_data'].attrs
            snapshots['istep'][i] = istep
            for name in scalar_data.keys():
                snapshots[name][i] = scalar_data[name]
    timer.record(f"Found {len(snapshots)} Dyablo snapshots in {path} with prefix {prefix}.")
    return snapshots

def read_cell(path, iout=None, istep=None, prefix=None):
    timer.start(f"Reading Dyablo cell data from {path} with prefix {prefix} and istep {istep}...")
    if istep is not None:
        if iout is not None:
            warnings.warn("Both iout and istep are provided. Ignoring iout and using istep to find the file.")
        if prefix is None:
            pattern = os.path.join(path, config['FILENAME_FORMAT_DYABLO'].format(prefix=ANY, istep=istep))
            files = glob.glob(pattern)
            if len(files) == 0:
                raise FileNotFoundError(f"No files found matching pattern {pattern}.")
            elif len(files) > 1:
                raise FileExistsError(f"Multiple files found matching pattern {pattern}. Please specify a prefix.")
            filename = files[0]
        else:
            filename = os.path.join(path, config['FILENAME_FORMAT_DYABLO'].format(prefix=prefix, istep=istep))
    elif iout is not None:
        if prefix is None:
            pattern = os.path.join(path, config['FILENAME_FORMAT_DYABLO'].format(prefix=ANY, istep=ANY))
        else:
            pattern = os.path.join(path, config['FILENAME_FORMAT_DYABLO'].format(prefix=prefix, istep=ANY))

        files = glob.glob(pattern)
        if len(files) == 0:
            raise FileNotFoundError(f"No files found matching pattern {pattern}.")
        files.sort()
        if iout > 0:
            iout -= 1
        elif iout == 0:
            raise ValueError("iout cannot be 0. It should be a positive or negative integer.")
        filename = files[iout]
    else:
        raise ValueError("Either iout or istep must be provided to find the file.")

    with h5py.File(filename, 'r') as f:
        connectivity = f['connectivity'][:]
        coordinates = f['coordinates'][:]
        n_vertices = connectivity.shape[1]
        vertices = np.reshape(coordinates[connectivity], newshape=(-1, n_vertices, 3))
        domain_length_max = np.max(np.max(coordinates, axis=0) - np.min(coordinates, axis=0))
        centers = np.mean(vertices, axis=1)
        dtypes = [('position_x', 'f8'), ('position_y', 'f8'), ('position_z', 'f8')]
        if 'level' not in f.keys():
            levels = np.round(-np.log2((vertices[:, 1, 0] - vertices[:, 0, 0]) / domain_length_max)).astype(int)
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
            table['level'] = levels

    timer.record(f"Finished reading Dyablo cell data from {filename}. Found {n_data} cells.")
    return table
