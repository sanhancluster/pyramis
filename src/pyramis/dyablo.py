from . import config
from .hdf import get_by_type
import h5py
import os
import numpy as np

def read_cell(path, istep, prefix=None):
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
