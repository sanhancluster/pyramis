import numpy as np
from . import config

def get_vname(vname: str, name_set: str | None=None):
    if name_set is not None:
        mapping = config['VNAME_MAPPING'][name_set]
    else:
        mapping = config['VNAME_MAPPING'][config['VNAME_SET']]
    vname = mapping.get(vname, vname)
    return vname


def get_mapping(name_set_from, name_set_to):
    mapping_to = config['VNAME_MAPPING'][name_set_to]
    if name_set_from == 'native':
        return mapping_to

    mapping_from = config['VNAME_MAPPING'][name_set_from]
    # Create reverse mapping from name_set_from
    reverse_from = {v: k for k, v in mapping_from.items() if isinstance(v, str)}

    # Create mapping from name_set_from to name_set_to
    mapping = {}
    for k, v in mapping_from.items():
        if isinstance(v, str):
            mapping[v] = mapping_to.get(k, k)
        else:
            mapping[k] = mapping_to.get(k, v)
    return mapping


def get_dim_keys(name_set=None):
        return get_vname('DIM_KEYS', name_set=name_set)


def get_vector(data, name_format: str='{key}', axis=-1) -> np.ndarray:
    return np.stack([data[f'{name_format.format(key=key)}'] for key in get_dim_keys()], axis=axis)


def get_position(data, axis=-1) -> np.ndarray:
    return get_vector(data, name_format='{key}', axis=axis)


def get_velocity(data, axis=-1) -> np.ndarray:
    return get_vector(data, name_format='v{key}', axis=axis)


def get_cell_size(data, boxlen: float=1.0):
    return boxlen * 2.**-data[get_vname('level')]


def uniform_digitize(values, lim, nbins):
    """
    A faster version of np.digitize that works with uniform bins.
    The result may vary from np.digitize near the bin edges.

    Parameters
    ----------
    values : array-like
        The input values to digitize.
    lim : array-like
        The limits for the bins.
    nbins : int
        The number of bins.

    Returns
    -------
    array-like
        The digitized indices of the input values.
    """
    values_idx = (values - lim[..., 0]) / (lim[..., 1] - lim[..., 0]) * nbins + 1
    values_idx = values_idx.astype(int)
    values_idx = np.clip(values_idx, 0, nbins+1)
    return values_idx
