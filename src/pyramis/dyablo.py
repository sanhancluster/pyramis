from fileinput import filename
import warnings

from . import config
from .hdf import get_by_type
import h5py
import os
import numpy as np
import glob
from . import ANY

def check_snapshots(path, prefix=None):
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

    return snapshots

def read_cell(path, istep=None, prefix=None):
    if istep is None:
        filename = path
    else:
        filename = os.path.join(path, config['FILENAME_FORMAT_DYABLO'].format(prefix=prefix, istep=istep))
    with h5py.File(filename, 'r') as f:
        connectivity = f['connectivity'][:]
        coordinates = f['coordinates'][:]
        n_vertices = connectivity.shape[1]
        vertices = np.reshape(coordinates[connectivity], newshape=(-1, n_vertices, 3))
        domain_length_max = np.max(np.max(coordinates, axis=0) - np.min(coordinates, axis=0))
        centers = np.mean(vertices, axis=1)
        levels = np.round(-np.log2((vertices[:, 1, 0] - vertices[:, 0, 0]) / domain_length_max)).astype(int)

        n_data = connectivity.shape[0]

        keys = [k for k in f.keys() if k not in ['connectivity', 'coordinates', 'scalar_data']]
        dtypes = [('position_x', 'f8'), ('position_y', 'f8'), ('position_z', 'f8'), ('level', 'i4')]
        
        for key in keys:
            dtypes.append((key, f[key].dtype))
        dtypes = np.dtype(dtypes)
        table = np.empty(n_data, dtype=dtypes)

        for key in keys:
            table[key] = f[key][:]

        table['position_x'] = centers[:, 0]
        table['position_y'] = centers[:, 1]
        table['position_z'] = centers[:, 2]
        table['level'] = levels

    return table
