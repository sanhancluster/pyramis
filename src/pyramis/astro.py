import numpy as np
from . import config, cgs_unit, cgs_constants, get_vname
from scipy.integrate import cumulative_trapezoid

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

    # build array in log aexp space

    x = np.linspace(np.log(aexp_min), np.log(aexp_max), nbins)
    aexp = np.exp(x)
    E = np.sqrt(omega_m * aexp**-3 + omega_l + omega_k * aexp**-2 + omega_r * aexp**-4)

    dtsc_over_dx = 1.0 / (aexp**2 * E)
    #dtsc_over_dx = np.exp(-x) / E
    tsc = cumulative_trapezoid(dtsc_over_dx, x, initial=0.0)
    tsc = tsc - np.interp(1.0, aexp, tsc)

    dt_over_dx = 1. / (H0 * cgs_unit.km / cgs_unit.Mpc * E)
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


def get_age(data, info, unit=None):
    t0 = cosmo_convert(info['cosmo_table'], data[get_vname('birth_time')], 't_sc', 'age')
    aexp = info['aexp']
    t = cosmo_convert(info['cosmo_table'], aexp, 'aexp', 'age')
    age = t - t0
    if unit is not None:
        age /= cgs_unit.__getattribute__(unit)
    return age


def get_temperature(data, info, unit='K'):
    unit_T = info.get('unit_t', 1.0) ** -2 * info.get('unit_l', 1.0) ** 2 / cgs_constants.k_B * cgs_constants.m_u
    temperature = data[get_vname('pressure')] / data[get_vname('density')] * unit_T / cgs_unit.__getattribute__(unit)

    return temperature
