import numpy as np
from . import config, cgs_unit
import re

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


def get_dim_keys():
        return ['x', 'y', 'z']


def get_vector_keys(name_format: str | None=None):
    if name_format is None:
        name_format = get_vname('POSITION_FORMAT')
    return [name_format.format(dim=dim) for dim in get_dim_keys()]


def get_vector(data, name_format: str | None=None, axis=-1) -> np.ndarray:
    return np.stack([data[name] for name in get_vector_keys(name_format=name_format)], axis=axis)


def get_position_keys():
    return get_vector_keys(name_format=get_vname('POSITION_FORMAT'))


def get_position(data, axis=-1) -> np.ndarray:
    return get_vector(data, name_format=get_vname('POSITION_FORMAT'), axis=axis)


def get_velocity_keys():
    return get_vector_keys(name_format=get_vname('VELOCITY_FORMAT'))


def get_velocity(data, axis=-1) -> np.ndarray:
    return get_vector(data, name_format=get_vname('VELOCITY_FORMAT'), axis=axis)


def get_cell_size(data, boxlen: float=1.0):
    return boxlen * 2.**-data[get_vname('level')]


def get_mass(data, boxlen: float=1.0):
    vname_mass = get_vname('mass')
    if vname_mass in data.dtype.names:
        return data[vname_mass]
    else:
        cell_size = get_cell_size(data, boxlen=boxlen)
        return data[get_vname('density')] * cell_size**3


def _parse_unit_string(unit_str):
    """
    Parse a unit string like 'km/s', 'g/cm^3', 'km/s^2' and return
    (cgs_factor, dim) where dim is a tuple of (L, M, T) exponents.

    cgs_unit: {'km': (1e5, (1,0,0)), ...}
    """

    def parse_tokens(expr):
        """Parse 'km^2', 'g', 's^-2' -> [('km', 2), ('g', 1), ('s', -2)]"""
        tokens = re.findall(r'([A-Za-z]+)(?:\^(-?\d+))?', expr)
        return [(name, int(exp) if exp else 1) for name, exp in tokens]

    # Split into numerator and denominator(s)
    # e.g. 'g/cm^3/s' -> numer='g', denoms=['cm^3', 's']
    parts = unit_str.split('/')
    numer = parts[0]
    denoms = parts[1:]

    cgs_factor = 1.0
    dim = [0, 0, 0]

    for name, exp in parse_tokens(numer):
        if name not in cgs_unit:
            raise ValueError(f"Unknown unit: '{name}'")
        factor = cgs_unit[name]['factor']
        d = cgs_unit[name]['dim']
        cgs_factor *= factor ** exp
        dim = [dim[i] + d[i] * exp for i in range(3)]

    for denom in denoms:
        for name, exp in parse_tokens(denom):
            if name not in cgs_unit:
                raise ValueError(f"Unknown unit: '{name}'")
            factor = cgs_unit[name]['factor']
            d = cgs_unit[name]['dim']
            cgs_factor /= factor ** exp
            dim = [dim[i] - d[i] * exp for i in range(3)]

    return cgs_factor, tuple(dim)


def get_unit(unit, info, aexp=None):
    """
    Get the conversion factor from code units to the specified unit string.
    if aexp is provided, it will be used to scale the length and time units according to the super-comoving reference frame.
    """
    unit_l = info.get('unit_l', 1.0)
    unit_m = info.get('unit_d', 1.0) * info.get('unit_l', 1.0) ** 3
    unit_t = info.get('unit_t', 1.0)

    if aexp is not None:
        aexp_factor = aexp / info['aexp']
        unit_l *= aexp_factor
        unit_t *= aexp_factor ** 2

    cgs_factor, dim = _parse_unit_string(unit)
    code_factor = cgs_factor / (unit_l ** dim[0] * unit_m ** dim[1] * unit_t ** dim[2])

    return code_factor



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


def format_bytes(bytes):
    if bytes < 1024:
        return f"{bytes:.2f} B"
    elif bytes < 1024**2:
        return f"{bytes / 1024:.2f} KiB"
    elif bytes < 1024**3:
        return f"{bytes / 1024**2:.2f} MiB"
    elif bytes < 1024**4:
        return f"{bytes / 1024**3:.2f} GiB"
    else:
        return f"{bytes / 1024**4:.2f} TiB"


class Wildcard:
    def __format__(self, format_spec):
        return '*'
    
    def __str__(self):
        return '*'
    
    def __eq__(self, other):
        return True

ANY = Wildcard()
