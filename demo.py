import pyramis.io as io
import pyramis.hdf as hdf
import pyramis.core as core
import numpy as np
import time
from rur import uri
from pyramis import get_cosmo_table, cosmo_convert
from matplotlib import pyplot as plt
import pyramis
import h5py
import os


if __name__ == '__main__':
    t = time.time()
    table = get_cosmo_table(70.0, 0.3, 0.7, nbins=5000, aexp_max=1.0)
    #print(table['ttilde'])
    print("Time elapsed:", time.time() - t)


    iout = 2
    path = '/home/hansan/simulation/ramses/v6h'
    region = [[0.0, 0.1], [0., 0.2], [0., 0.2]]
    t = time.time()

    t = time.time()
    print(io.get_available_snapshots(path, scheduled_only=True))
    print("Time elapsed:", time.time() - t)

    cell = hdf.read_cell(path='/home/hansan/simulation/ramses/v6h/hdf', iout=1, region=[[0.0, 0.1], [0., 0.2], [0., 0.2]], use_process=False, n_workers=4)
    print(cell.size)

    cell = hdf.read_cell(path='/home/hansan/simulation/ramses/v6h/hdf', iout=1, region=[[0.0, 0.1], [0., 0.2], [0., 0.2]], use_process=False, n_workers=1)
    print(cell.size)


    part = hdf.read_part(path='/home/hansan/simulation/ramses/v6h/hdf', iout=1, part_type='dm', region=[[0.0, 0.1], [0., 0.2], [0., 0.2]], use_process=False, n_workers=1)
    print(part.size)
    print(part.dtype)

    #f = h5py.File(os.path.join(path, 'hdf/', 'cell_00001.h5'))
    #print(f['leaf']['data'][:])


    #cell = hdf.read_cell(path='/home/hansan/simulation/ramses/v6h/hdf', iout=1, region=[[0.0, 0.1], [0., 0.2], [0., 0.2]], #use_process=True, n_workers=2)
    #print(cell.size)
