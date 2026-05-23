

import os

from pyramis.config_module import get_config, get_vname
from .ramses import parse_info, parse_namelist
from .utils.arrayview import ArrayView
from .utils.fortranfile import FortranFile
import numpy as np
from . import get_dim_keys, get_position_names

def read_info(path, iout):
    path_info = os.path.join(path, f"output_{iout:05d}", f"info.txt")
    info = parse_info(path_info)
    return info


def parse_hydro_names(filename):
    names = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("variable #"):
                name = line.split(":")[1].strip()
                names.append(name)
    return names


def read_cell(path, iout, read_hydro=True, read_branch=False, n_workers=1, target_fields=None, namelist_path=None):
    config = get_config()
    info = read_info(path, iout)

    nfile = info['nfile']

    levelmin = info['levelmin']
    nlevelmax = info['levelmax']
    ndim = info['ndim']
    numbl = np.zeros([nlevelmax, nfile], dtype=np.int32)
    ngrid = np.zeros(nlevelmax, dtype=np.int32)

    format_position = 'f4'
    format_hydro = 'f4'

    amr_nvar = ndim + 2**ndim
    twotondim = 2**ndim
    dtype_out = []
    dtype_out += [(idim, format_position) for idim in get_position_names(ndim)]
    dtype_out += [('level', 'i2')]

    all_hydro_names = []
    hydro_names = []

    if namelist_path is None:
        namelist_path = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout), config['NAMELIST_FILENAME'])
    nml = parse_namelist(namelist_path)
    bound_levelmin = nml.get('BOUNDARY_PARAMS', {}).get('bound_levelmin', 0)
    bound_levelmin = int(bound_levelmin)
    box_xmin = nml.get('BOUNDARY_PARAMS', {}).get('box_xmin', 0)
    box_xmax = nml.get('BOUNDARY_PARAMS', {}).get('box_xmax', 0)
    box_ymin = nml.get('BOUNDARY_PARAMS', {}).get('box_ymin', 0)
    box_ymax = nml.get('BOUNDARY_PARAMS', {}).get('box_ymax', 0)
    box_zmin = nml.get('BOUNDARY_PARAMS', {}).get('box_zmin', 0)
    box_zmax = nml.get('BOUNDARY_PARAMS', {}).get('box_zmax', 0)
    bound_idx = np.array([[box_xmin, box_xmax], [box_ymin, box_ymax], [box_zmin, box_zmax]], dtype=np.int32)[:ndim]
    domain_lims = np.array([[0, 1]] * ndim)

    if read_hydro:
        all_hydro_names = parse_hydro_names(os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout), 'hydro_header.txt'))
        hydro_names = all_hydro_names
        dtype_out += [(name, format_hydro) for name in all_hydro_names]

    if target_fields is not None:
        target_fields = set(target_fields)
        dtype_out = [field for field in dtype_out if field[0] in target_fields]
        hydro_names = [name for name in all_hydro_names if name in target_fields]
    
    dtype_out = np.dtype([(get_vname(name), dtype) for name, dtype in dtype_out])

    # Precompute cell bit offsets from grid center (matching RAMSES xc convention)
    cell_bits = np.zeros((twotondim, ndim), dtype=np.int32)
    inds = np.arange(twotondim)
    iz = inds // 4 if ndim > 2 else 0
    iy = (inds - 4 * iz) // 2 if ndim > 1 else 0
    ix = inds - 2 * iy - 4 * iz
    cell_bits = np.array([ix, iy, iz][:ndim])

    # AMR counting loop (matches rd_amr: skip=12, nvar=ndim+2**ndim)
    skip = 12
    for ifile in range(1, nfile + 1):
        filename = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout), config['FILENAME_FORMAT_MINIRAMSES'].format(data='amr', ifile=ifile))
        for ilevel in range(levelmin-1, nlevelmax):
            offset = skip + 4 * (ilevel + 1 - levelmin)
            numbl[ilevel, ifile-1] = np.fromfile(filename, dtype=np.int32, count=1, offset=offset)[0]
            ngrid[ilevel] += numbl[ilevel, ifile-1]

    ngrid_total = ngrid.sum()
    ncell_total = ngrid_total * twotondim

    out = np.empty(ncell_total, dtype=dtype_out)
    idx_read = np.zeros(ncell_total, dtype=int)
    nread_arr = np.zeros((nlevelmax, nfile), dtype=int)

    # AMR reading loop: expand each grid to ntwotondim cell positions (matches rd_amr)
    iskip = 0
    for ifile in range(1, nfile + 1):
        filename = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout), config['FILENAME_FORMAT_MINIRAMSES'].format(data='amr', ifile=ifile))
        offset = 12 + 4 * (nlevelmax + 1 - levelmin)
        for ilevel in range(levelmin, nlevelmax+1):
            ncache = numbl[ilevel-1, ifile-1]

            transfer = np.fromfile(filename, dtype=np.int32, count=amr_nvar*ncache, offset=offset)
            transfer = np.reshape(transfer, (ncache, amr_nvar))

            refined = transfer[:, ndim:ndim+twotondim]
            to_read = (refined == 0) if not read_branch else (refined != 0)

            iread, jread = np.nonzero(to_read)
            nread = len(iread)
            idx_read[iskip:iskip + nread] = iread * twotondim + jread
            nread_arr[ilevel-1, ifile-1] += nread
            # xx(i,idim)=(2*m%grid(igrid+i-1)%ckey(idim)+MOD((ind-1)/nstride,2)+0.5)*dx-m%skip(idim)
            dx = 1./2**ilevel
            cell_pos = (2*transfer[:, :ndim, None]+cell_bits+0.5) * dx
            if bound_levelmin > 0:
                bound_grid = np.arange(2**(bound_levelmin-1)) * 2.**-(bound_levelmin-1)
                bound = bound_grid[bound_idx]
                cell_pos = (cell_pos - bound[:, 0, None]) / (bound[:, 1, None] - bound[:, 0, None])
                domain_lims = (domain_lims - bound[:, 0, None]) / (bound[:, 1, None] - bound[:, 0, None])

            for idim, key in enumerate(get_position_names(ndim)):
                out[iskip:iskip + nread][get_vname(key)] = cell_pos[iread, idim, jread]
            out[iskip:iskip + nread][get_vname('level')] = ilevel

            offset += ncache * amr_nvar * 4
            iskip += nread
    out = out[:iskip]
    idx_read = idx_read[:iskip]

    if read_hydro:
        # Hydro counting loop (matches rd_hydro: skip=16)
        skip = 16
        nhvar_file = len(all_hydro_names)
        nvartot = nhvar_file * twotondim
        ngrid = np.zeros(nlevelmax, dtype=np.int32)

        for ifile in range(1, nfile + 1):
            filename = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout), config['FILENAME_FORMAT_MINIRAMSES'].format(data='hydro', ifile=ifile))
            for ilevel in range(levelmin, nlevelmax+1):
                offset = skip + 4 * (ilevel - levelmin)
                numbl[ilevel-1, ifile-1] = np.fromfile(filename, dtype=np.int32, count=1, offset=offset)[0]
                ngrid[ilevel-1] += numbl[ilevel-1, ifile-1]

        # Hydro reading loop (matches rd_hydro: offset reset per file, grid-first cell ordering)
        iskip = 0
        for ifile in range(1, nfile + 1):
            filename = os.path.join(path, config['OUTPUT_FORMAT'].format(iout=iout), config['FILENAME_FORMAT_MINIRAMSES'].format(data='hydro', ifile=ifile))
            offset = 16 + 4 * (nlevelmax + 1 - levelmin)
            for ilevel in range(levelmin, nlevelmax+1):
                ncache = numbl[ilevel-1, ifile-1]
                nread_level = nread_arr[ilevel-1, ifile-1]
                idx_level = idx_read[iskip:iskip + nread_level]
                iread_level = idx_level // twotondim
                jread_level = idx_level % twotondim

                transfer = np.fromfile(filename, dtype=format_hydro, count=nvartot * ncache, offset=offset)
                transfer = np.reshape(transfer, (ncache, nhvar_file, twotondim))
                transfer = np.transpose(transfer, (1, 2, 0))  # (nhvar_file, ntwotondim, ncache)
                data = transfer[:, jread_level, iread_level]  # (nread_level, nhvar_file)

                for name in hydro_names:
                    ivar = all_hydro_names.index(name)
                    # .T gives (ncache, ntwotondim); reshape(-1) flattens grid-first
                    out[iskip:iskip + nread_level][get_vname(name)] = data[ivar]

                offset += ncache * nvartot * 4
                iskip += nread_level

    info['domain_lims'] = domain_lims

    return ArrayView(out, info=info)
