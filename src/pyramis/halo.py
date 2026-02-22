import os
import pickle as pkl

from . import config, get_vname
from .utils.fortranfile import FortranFile
import numpy as np

def _get_halomaker_struct(galaxy=False, double_precision=True, nbin:int=10, vname_set=config['VNAME_SET']):
    ftype = 'f8' if double_precision else 'f4'
    base = [
        ('num_parts', 'i4'), None, ('identity', 'i4'), ('timestep', 'i4'),
        (['level', 'hosthalo', 'hostsub', 'nbsub', 'nextsub'], 'i4'),
        ('mass', ftype), (['position_x', 'position_y', 'position_z'], ftype),
        (['velocity_x', 'velocity_y', 'velocity_z'], ftype),
        (['angular_momentum_x', 'angular_momentum_y', 'angular_momentum_z'], ftype),
        (['radius', 'semi_axis_a', 'semi_axis_b', 'semi_axis_c'], ftype),
        (['kinetic_energy', 'potential_energy', 'total_energy'], ftype), ('spin_parameter', ftype),
    ]
    if galaxy:
        base += [(['sigma', 'sigma_bulge', 'mass_bulge'], ftype)]
    else:
        if double_precision:
            base += [('sigma', ftype)]
    base += [
        (['virial_radius', 'virial_mass', 'virial_temperature', 'circular_velocity'], ftype),
        (['density_0', 'radius_c'], ftype)
    ]
    if galaxy:
        base += [('n_bin', 'i4'), ('r_bin', ftype, nbin), ('density_profile', ftype, nbin)]

    return base


def _read_halo(f:FortranFile, data, struct, vname_set=config['VNAME_SET']):
    for item in struct:
        if item is None:
            f.skip_records(1)
        else:
            read = f.read_record('b')
            names, dtype, *shape = item
            if not isinstance(names, list):
                names = [names]
            names = [get_vname(name, vname_set) for name in names]
            read = np.array(read).view(dtype)
            if len(shape) > 0:
                read = read.reshape((-1, *shape))
            for name, value in zip(names, read):
                data[name] = value


def read_halomaker(path: str, iout:int | None=None, galaxy=False, double_precision: bool=True, vname_set=config['VNAME_SET']):
    if iout is not None:
        if galaxy:
            path = os.path.join(path, config['FILENAME_FORMAT_GALAXYMAKER'].format(iout=iout))
        else:
            path = os.path.join(path, config['FILENAME_FORMAT_HALOMAKER'].format(iout=iout))
    
    f = FortranFile(path, 'r')
    nbodies = f.read_ints('i4')
    massp = f.read_reals('f4')
    if double_precision:
        aexp = f.read_reals('f8')
    else:
        aexp = f.read_reals('f4')
    
    omega_t = f.read_reals('f4')
    age_univ = f.read_reals('f4')
    nb_of_halos, nb_of_subhalos = f.read_ints('i4')
    nhalo_snap = nb_of_halos + nb_of_subhalos
    nbin = 0
    if galaxy:
        f.skip_records(15)
        nbin, = f.read_ints('i4')
        f = FortranFile(path, 'r')
        f.skip_records(6)
        
    struct = _get_halomaker_struct(galaxy=galaxy, double_precision=double_precision, nbin=nbin)
    dtypes = []
    for item in struct:
        if item is not None:
            names, dtype, *shape = item
            if not isinstance(names, list):
                names = [names]
            for name in names:
                if len(shape) > 0:
                    dtypes.append((name, dtype, *shape))
                else:
                    dtypes.append((name, dtype))
    data = np.zeros(nhalo_snap, dtype=dtypes)

    for i in range(nhalo_snap):
        _read_halo(f, data[i], struct, vname_set=vname_set)
    return data


def read_ptree(path: str, iout:int | None=None, all=False):
    if iout is not None:
        path = os.path.join(path, config['FILENAME_FORMAT_PTREE'].format(iout=iout))
    return pkl.load(open(path, "rb"))