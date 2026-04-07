import numpy as np
import pyramis as pyr
import h5py
import argparse
import time
from pyramis.utils import Timestamp
import os

pyr.set_config('VNAME_SET', 'native')
timer = Timestamp()

def export_hdf(repo: str, output_path='SINKPROPS/sinkprops.h5', h5py_kwargs=None):
    if h5py_kwargs is None:
        h5py_kwargs = dict(compression='lzf', chunks=True, shuffle=True)

    timer.start("Reading sink properties from RAMSES snapshots")
    sp = pyr.ramses.read_sinkprops(repo, use_process=True, copy_result=True)
    timer.record("Finished reading sink properties from RAMSES snapshots")

    timer.start("Processing sink properties")
    icoarses, indices, counts = np.unique(sp[pyr.get_vname('icoarse')], return_index=True, return_counts=True)
    steps = np.empty(len(icoarses), dtype=[(pyr.get_vname('icoarse'), 'i4'), (pyr.get_vname('aexp'), 'f8'), (pyr.get_vname('unit_l'), 'f4'), (pyr.get_vname('unit_d'), 'f4'), (pyr.get_vname('unit_t'), 'f4'), (pyr.get_vname('num'), 'i4'), (pyr.get_vname('offset'), 'i4')])

    for name in ['icoarse', 'aexp', 'unit_l', 'unit_d', 'unit_t']:
        steps[pyr.get_vname(name)] = sp[indices][pyr.get_vname(name)]
    steps[pyr.get_vname('num')] = counts
    steps[pyr.get_vname('offset')] = indices
    # sort the data by id
    id_key = np.argsort(sp, order=[pyr.get_vname('identity'), pyr.get_vname('icoarse')], kind='stable')
    sp = sp[id_key]
    icoarse_key = np.argsort(id_key)

    ids, indices, counts = np.unique(sp[pyr.get_vname('identity')], return_index=True, return_counts=True)
    sinks = np.empty(len(ids), dtype=[(pyr.get_vname('identity'), 'i4'), (pyr.get_vname('num'), 'i4'), (pyr.get_vname('offset'), 'i4'), (pyr.get_vname('icoarse_min'), 'i4'), (pyr.get_vname('icoarse_max'), 'i4')])
    sinks[pyr.get_vname('identity')] = ids
    sinks[pyr.get_vname('num')] = counts
    sinks[pyr.get_vname('offset')] = indices
    sinks[pyr.get_vname('icoarse_min')] = np.array([sp[i:i+n][pyr.get_vname('icoarse')].min() for i, n in zip(indices, counts)])
    sinks[pyr.get_vname('icoarse_max')] = np.array([sp[i:i+n][pyr.get_vname('icoarse')].max() for i, n in zip(indices, counts)])

    descr_new = []
    for name in sp.dtype.names:
        if name in [pyr.get_vname('aexp'), pyr.get_vname('position_x'), pyr.get_vname('position_y'), pyr.get_vname('position_z')]:
            descr_new.append((name, 'f8'))
        else:
            descr_new.append((name, sp.dtype.fields[name][0]))
    sp_new = np.empty(sp.shape, dtype=descr_new)
    for name in sp.dtype.names:
        sp_new[name] = sp[name]
    sp = sp_new
    
    timer.record("Finished processing sink properties")

    timer.start("Writing sink properties to HDF5")
    write_path = os.path.join(repo, output_path)
    with h5py.File(write_path, 'w') as f:
        f.create_dataset('steps', data=steps, **h5py_kwargs)
        f.create_dataset('sinks', data=sinks, **h5py_kwargs)
        f.create_dataset('data', data=sp, **h5py_kwargs)
        f.create_dataset('icoarse_key', data=icoarse_key, **h5py_kwargs)

        f.attrs['description'] = "Sink properties exported from RAMSES snapshots. 'steps' dataset contains one entry per snapshot with sink particles, while 'sinks' dataset contains one entry per sink particle. The 'data' dataset contains the original sink properties for all sink particles, sorted by their identity. The 'offset' field in 'sinks' indicates the starting index of each sink's properties in the 'data' dataset, and the 'num' field indicates how many entries belong to that sink."
        f.attrs['vname_set'] = pyr.config['VNAME_SET']
        f.attrs['created'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        f.attrs['n_step'] = len(steps)
        f.attrs['n_sink'] = len(sinks)
        f.attrs['n_data'] = len(sp)

        f.attrs[f"{pyr.get_vname('icoarse')}_max"] = np.max(steps[pyr.get_vname('icoarse')])
        f.attrs[f"{pyr.get_vname('identity')}_max"] = np.max(sinks[pyr.get_vname('identity')])


    timer.record("Finished writing sink properties to HDF5")

def main():
    parser = argparse.ArgumentParser(description="Export sink properties to HDF5.")
    parser.add_argument("repo", type=str, help="Path to the repository.")
    parser.add_argument("--output", type=str, default="SINKPROPS/sinkprops.h5", help="Output HDF5 file path.")
    parser.add_argument("--compression", type=str, default="lzf", help="Compression method for HDF5 datasets.")
    args = parser.parse_args()

    h5py_kwargs = dict(compression=args.compression, chunks=True, shuffle=True)
    print(f"Exporting sink properties from {args.repo} to {args.output} with compression={args.compression}...")

    timer.start("Exporting sink properties to HDF5", name='main')
    export_hdf(args.repo, output_path=args.output, h5py_kwargs=h5py_kwargs)
    timer.record("Finished exporting sink properties to HDF5", name='main')

    print("Export completed.")

if __name__ == '__main__':
    main()