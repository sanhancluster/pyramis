import os
import pickle as pkl

from . import config, get_vname, timer, format_bytes
from .utils.fortranfile import FortranFile
import numpy as np
from typing import Sequence

def _get_halomaker_struct(galaxy=False, double_precision=True, nbin:int=10):
    """
    Get the structured array dtype for reading single halo data.
    """
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


def _get_halomaker_skip(galaxy=False, double_precision=True):
    """
    Get the number of records to skip for reading single halo data.
    """
    struct = _get_halomaker_struct(galaxy=galaxy, double_precision=double_precision)
    return len(struct)


def _read_halo(f:FortranFile, data, struct, vname_set='native'):
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


def read_halomaker(
        path: str,
        iout:int | None=None,
        galaxy=False,
        double_precision: bool=True,
        vname_set=None,
        error_on_missing=False) -> np.ndarray:
    """
    Read HaloMaker output data.

    Parameters
    ----------
    path : str
        Path to the directory of the repository.
    iout : int or None, optional
        Output number to read. If None, the path is used directly. Default is None.
    galaxy : bool, optional
        Whether to read galaxy data. Default is False.
    double_precision : bool, optional
        Whether to read data in double precision. Default is True.
    vname_set : str or None, optional
        Variable name set to use. If None, the default from config is used. Default is None.
    error_on_missing : bool, optional
        Whether to raise an error if the file is missing. Default is False.
    """

    if vname_set is None:
        vname_set = config['VNAME_SET']

    if iout is not None:
        if galaxy:
            path = os.path.join(path, config['FILENAME_FORMAT_GALAXYMAKER'].format(iout=iout))
        else:
            path = os.path.join(path, config['FILENAME_FORMAT_HALOMAKER'].format(iout=iout))
    
    timer.start(f"Reading halo data from {path} at iout={iout}...")

    if not os.path.exists(path):
        if error_on_missing:
            raise FileNotFoundError(f"File {path} not found.")
        else:
            timer.record(f"File {path} not found. Skipping reading halo data at iout={iout}.")
            nbin = 10
            nhalo_snap = 0
    else:
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

    struct = _get_halomaker_struct(galaxy=galaxy, double_precision=double_precision, nbin=nbin)
    dtypes = []
    for item in struct:
        if item is not None:
            names, dtype, *shape = item
            if not isinstance(names, list):
                names = [names]
            names = [get_vname(name, vname_set) for name in names]
            for name in names:                
                if len(shape) > 0:
                    dtypes.append((name, dtype, *shape))
                else:
                    dtypes.append((name, dtype))
    data = np.zeros(nhalo_snap, dtype=dtypes)

    if nhalo_snap == 0:
        return data
    else:
        f = FortranFile(path, 'r')
        f.skip_records(6)
        for i in range(nhalo_snap):
            _read_halo(f, data[i], struct, vname_set=vname_set)
        f.close()

    size_byte = data.nbytes
    timer.record(f"Finished reading {nhalo_snap} halo data ({format_bytes(size_byte)}) from {path}.")

    return data


def read_halomaker_members(
        path:str,
        iout:int | None=None,
        target_id:int | Sequence[int] | np.ndarray | None=None,
        galaxy=False,
        double_precision: bool=True,
        return_nparts=False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Reads the particle IDs of the halos from HaloMaker output.

    Parameters
    ----------
    path : str
        Path to the directory of the repository.
    iout : int or None, optional
        Output number to read. If None, the path is used directly. Default is None.
    target_id : int, Sequence[int], np.ndarray, or None, optional
        Halo ID(s) to read. If None, all halos are read. Default is None.
    galaxy : bool, optional
        Whether to read galaxy data. Default is False.
    double_precision : bool, optional
        Whether to read data in double precision. Default is True.
    return_nparts : bool, optional
        Whether to return the number of particles for each halo. Default is False.
    """

    nskip = _get_halomaker_skip(galaxy=galaxy, double_precision=double_precision)

    if target_id is not None:
        target_id = np.atleast_1d(target_id)

    if iout is not None:
        if galaxy:
            path = os.path.join(path, config['FILENAME_FORMAT_GALAXYMAKER'].format(iout=iout))
        else:
            path = os.path.join(path, config['FILENAME_FORMAT_HALOMAKER'].format(iout=iout))

    timer.start(f"Reading halo member data from {path}...")

    # Get the number of particles for each halo
    with FortranFile(path, 'r') as f:
        f.skip_records(5)
        nb_of_halos, nb_of_subhalos = f.read_ints('i4')
        nhalo_snap = nb_of_halos + nb_of_subhalos

        nparts_arr = []
        idlist_all = []
        for ihalo in range(1, nhalo_snap+1):
            nparts = f.read_ints()[0]
            nparts_arr.append(nparts)
            f.skip_records(1)
            idlist_all.append(f.read_ints()[0])
            f.skip_records(nskip - 3) # skip the rest of the halo data
    idlist_all = np.array(idlist_all)

    # Build the output array and offsets for reading the particle IDs
    nparts_arr = np.array(nparts_arr)
    if target_id is not None:
        # idlist = idlist[np.isin(idlist, idlist_all)]
        indices = np.searchsorted(idlist_all, target_id)
        nparts_arr = np.array([(nparts_arr[idx] if id in idlist_all else 0) for idx, id in zip(indices, target_id)])

    offsets_out = np.concatenate(([0], np.cumsum(nparts_arr[:-1])))
    nparts_out = np.sum(nparts_arr)
    members = np.empty(nparts_out, dtype='i4')
    
    # Read the particle IDs for the specified halos
    with FortranFile(path, 'r') as f:
        f.skip_records(6)
        for ihalo in range(1, nhalo_snap+1):
            if target_id is None or ihalo in target_id:
                idx = np.where(target_id == ihalo)[0][0] if target_id is not None else ihalo - 1
                f.skip_records(1) # skip nparts
                read = f.read_ints() # read members
                members[offsets_out[idx]:offsets_out[idx]+nparts_arr[idx]] = read
                f.skip_records(nskip - 2)
            else:
                f.skip_records(nskip)
    num_target = len(target_id) if target_id is not None else nhalo_snap
    timer.record(f"Found {nparts_out} members for {num_target} halos in total.")

    return members if not return_nparts else (members, nparts_arr)


def read_ptree(path: str, iout:int | None=None):
    if iout is not None:
        path = os.path.join(path, config['FILENAME_FORMAT_PTREE'].format(iout=iout))
    return pkl.load(open(path, "rb"))