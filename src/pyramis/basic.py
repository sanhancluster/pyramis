import numpy as np
from . import get_config
from scipy.integrate import cumulative_trapezoid
from types import SimpleNamespace

config = get_config()
cgs_unit = SimpleNamespace(**config['CGS_UNIT'])

def get_vname(name: str, name_group: str | None=None):
    if name_group is not None:
        mapping = config['VNAME_MAPPING'][name_group]
    else:
        mapping = config['VNAME_MAPPING'][config['VNAME_GROUP']]
    vname = mapping.get(name, name)
    if vname is None:
        vname = name
    return vname


def get_dim_keys(name_group=None):
        return get_vname('DIM_KEYS', name_group=name_group)


def get_vector(data, name_format: str='{key}', axis=-1) -> np.ndarray:
    return np.stack([data[f'{name_format.format(key=key)}'] for key in get_dim_keys()], axis=axis)


def get_position(data, axis=-1) -> np.ndarray:
    return get_vector(data, name_format='{key}', axis=axis)


def get_velocity(data, axis=-1) -> np.ndarray:
    return get_vector(data, name_format='v{key}', axis=axis)


def get_cell_size(data, boxsize: float=1.0):
    return boxsize * 2.**-data[get_vname('level')]


def get_cosmo_table(H0: float, omega_m: float, omega_l: float, omega_k=None, omega_r=None, nbins=5000, aexp_min=1E-4, aexp_max=10.0) -> np.ndarray:
    """
    Build a conversion table for aexp, ttilde, and age of the universe.
    ttilde refers `conformal time (super-comoving time)` scale that is used in cosmological simulation in ramses.
    
    Parameters
    ----------
    H0 : float
        Hubble constant at z=0 in km/s/Mpc.
    omega_m : float
        Matter density parameter at z=0.
    omega_l : float
        Dark energy density parameter at z=0.
    nbins : int, optional
        Number of bins in the table, by default 5000.
    """
    def E(aexp):
        return np.sqrt(omega_m * aexp ** -3 + omega_l)
    
    if omega_r is None:
        omega_r = 0.0
    
    if omega_k is None:
        omega_k = 1.0 - omega_m - omega_l - omega_r

    x = np.linspace(np.log(aexp_min), np.log(aexp_max), nbins)
    aexp = np.exp(x)
    E = np.sqrt(omega_m * aexp**-3 + omega_l + omega_k * aexp**-2 + omega_r * aexp**-4)

    dtsc_over_dx = np.exp(-x) / E
    tsc = cumulative_trapezoid(dtsc_over_dx, x, initial=0.0)
    tsc = tsc - np.interp(1.0, aexp, tsc)

    dt_over_dx = 1. / (H0 * cgs_unit.km / cgs_unit.Mpc * E * cgs_unit.Gyr)
    age = cumulative_trapezoid(dt_over_dx, x, initial=0.0)
    z = 1.0 / aexp - 1.0
    table = np.rec.fromarrays([aexp, tsc, age, z], dtype=[('aexp', 'f8'), ('t_sc', 'f8'), ('age', 'f8'), ('z', 'f8')])

    return table


def cosmo_convert(table, x, xname, yname):
    x_arr = table[xname]
    y_arr = table[yname]

    if np.any(x < x_arr[0]) or np.any(x > x_arr[-1]):
        raise ValueError(f"{xname} out of bounds: valid range [{x_arr[0]}, {x_arr[-1]}]")

    y = np.interp(x, x_arr, y_arr)
    return y


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