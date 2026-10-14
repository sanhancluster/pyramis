# Pyramis: Python Reader for Adaptive Mesh Interface Simulations
![PyPI - Version](https://img.shields.io/pypi/v/pyramis)

A Python library that provides an essential, flexible and efficient framework for managing and analysing adaptive mesh simulations such as [RAMSES](https://github.com/ramses-organisation/ramses) and [Dyablo](https://github.com/Dyablo-HPC/Dyablo).

## Table of contents
- [Installing](#installing)
  - [Using pip](#using-pip)
  - [Using development mode](#using-development-mode)
  - [Using conda](#using-conda)
- [How to use](#how-to-use)
  - [Reading the RAMSES raw data](#reading-the-ramses-raw-data)
    - [The particle data](#the-particle-data)
    - [The cell data](#the-cell-data)
  - [Accessing fields and units](#accessing-fields-and-units)
  - [Reading the RAMSES HDF format data](#reading-the-ramses-hdf-format-data)
  - [Reading the Dyablo data](#reading-the-dyablo-data)
  - [Reading the HaloMaker data](#reading-the-halomaker-data)
  - [Drawing the AMR and particle data](#drawing-the-amr-and-particle-data)
    - [Simple plots with `vis`](#simple-plots-with-vis)
    - [Array output with `im`](#array-output-with-im)

## Installing
pyramis requires Python 3.11 or later.

### Using pip
```bash
pip install pyramis
```
### Using development mode
```bash
git clone https://github.com/sanhancluster/pyramis
cd pyramis

pip install -e .
```
### Using conda
To create a new conda environment specifically for pyramis:
```bash
git clone https://github.com/sanhancluster/pyramis
cd pyramis

conda env create -f environment.yml
conda activate pyramis

python -m pip install -e .
```
To install into an existing environment instead:
```bash
git clone https://github.com/sanhancluster/pyramis
cd pyramis

conda activate <my_env_name>
conda env update -f environment.yml

python -m pip install -e .
```

## How to use
### Reading the RAMSES raw data
pyramis reads RAMSES snapshot files in parallel, using threads (`concurrent.futures.ThreadPoolExecutor`) by default. The number of workers is set from the available CPU cores when the package is imported.

#### The particle data
`read_part` reads the particles inside a given region. pyramis uses the Peano-Hilbert space-filling curve to work out which CPU domains overlap the region, and reads only those files.
```python
import numpy as np
import pyramis as pr

# path to the directory where output_* are located
ramses_path = '/path/to/ramses/'
# box to read, in code units
region = [[0.4, 0.6], [0.4, 0.6], [0.4, 0.6]]

part = pr.ramses.read_part(ramses_path, iout=1, region=region)  # reads output_00001

print(f"Total particle mass within the box is {np.sum(part['mass'])}")  # in code units
```
- `part_type` selects one kind of particle: `'dm'`, `'star'`, `'cloud'` or `'tracer'`.
- `iout` can be negative to count back from the latest snapshot, e.g. `iout=-1` reads the last one.
- `target_fields` reads only the listed fields, which saves memory, e.g. `target_fields=['position_x', 'position_y', 'position_z', 'mass']`.
- `n_workers` overrides the number of workers, and `use_process=True` switches from threads to processes.
- `exact_cut=False` keeps every particle in the CPU domains that were read, instead of cutting exactly at the region boundary.
```python
star = pr.ramses.read_part(ramses_path, iout=1, region=region, part_type='star')
```

#### The cell data
`read_cell` reads the leaf cells inside a given region in the same way.
```python
cell = pr.ramses.read_cell(ramses_path, iout=1, region=region)

print(f"Mean gas density within the box is {np.mean(cell['density'])}")  # in code units
```
- `levelmax` limits the refinement level to read.
- `read_hydro=False` reads the AMR structure only, and `read_grav=True` also reads the gravity files.
- `target_fields`, `n_workers`, `use_process` and `exact_cut` work as for particles.

Sink particles are read with `pr.ramses.read_sink(ramses_path, iout=1)`.

### Accessing fields and units
The returned objects behave like numpy structured arrays. Besides the stored fields, a few derived fields are available by name:
- `'position'` and `'velocity'` return an (N, 3) array.
- `'mass'` on cells returns the cell mass computed from density and cell size, and `'cell_size'` returns the cell width.
- `'age'` (stars, in cosmological runs) is computed from the snapshot info, and `'temperature'` (cells) is returned in kelvin.

Any field can be converted to physical units by giving a unit string as the second key:
```python
x_kpc = part['position', 'kpc']
m_sun = part['mass', 'Msun']
rho = cell['density', 'Msun/pc^3']
```
Unit strings combine the units defined in the configuration (e.g. `cm`, `pc`, `kpc`, `Mpc`, `g`, `Msun`, `s`, `yr`, `Myr`, `Gyr`, `kps`, `erg`, `keV`) with `/` and `^`.

Field names follow the `'native'` naming set by default (`mass`, `density`, `position_x`, ...). To use short names (`m`, `rho`, `x`, ...) instead, call `pr.set_config('VNAME_SET', 'simple')` before reading.

### Reading the RAMSES HDF format data
RAMSES outputs can be converted into chunked HDF5 files with the `amr2hdf` command, which is installed with the package. The HDF files can then be read region by region without going through the raw RAMSES files.
```bash
amr2hdf --repo /path/to/ramses/ --output hdf --nthread 8
```
`--imin`/`--imax` or `--iouts` choose which snapshots to convert, and `--part`/`--cell` convert only one kind of data.

pyramis reads HDF files in parallel using processes (`concurrent.futures.ProcessPoolExecutor`) by default, so the top-level script needs an `if __name__ == '__main__':` guard.
```python
import pyramis as pr

if __name__ == '__main__':
    # path to the directory where part_*.h5, cell_*.h5 are located
    hdf_path = '/path/to/hdf/'
    # box to read, in code units
    region = [[0.4, 0.6], [0.4, 0.6], [0.4, 0.6]]

    part = pr.hdf.read_part(hdf_path, iout=1, region=region, part_type='star')
    cell = pr.hdf.read_cell(hdf_path, iout=1, region=region)
```
- If `part_type` is omitted, all particle types are read and merged into the fields they share.
- `levelmax_load` (cells) reads cells only up to the given refinement level.
- `subsample=n` reads every n-th element, for a quick look at large data.
- `use_process=False` switches to threads, which removes the need for the main guard.
- `target_fields`, `n_workers` and `exact_cut` work as for the raw RAMSES readers.

### Reading the Dyablo data
`read_cell` and `read_part` read a Dyablo snapshot, selected by output number (`iout`) or by step number (`istep`).
```python
dyablo_path = '/path/to/dyablo/'

cell = pr.dyablo.read_cell(dyablo_path, istep=0)
part = pr.dyablo.read_part(dyablo_path, iout=1)
```
If the directory holds runs with different file prefixes, pick one with `prefix`.

### Reading the HaloMaker data

#### DM Halo and Galaxy Catalog
`read_halomaker` reads a HaloMaker or GalaxyMaker `tree_bricks` catalogue.
```python
halomaker_path = '/path/to/halos/'  # path to the directory where HaloMaker tree_bricks* are located
galaxymaker_path = '/path/to/galaxies/'  # path to the directory where GalaxyMaker tree_bricks* are located

halo = pr.halo_finder.read_halomaker(halomaker_path, iout=1)
galaxy = pr.halo_finder.read_halomaker(galaxymaker_path, iout=1, galaxy=True)
```
- Positions are converted to code units and masses to solar masses by default. Set `position_to_code=False` or `mass_to_solar=False` to keep the original HaloMaker units.
- `double_precision=False` reads catalogues written in single precision.

### Drawing the AMR and particle data
pyramis provides two levels of drawing tools.
- `pr.vis` (`pyramis.visualize`) draws projected maps directly onto a matplotlib axis, for quick plots.
- `pr.im` (`pyramis.image`) returns the projected map as a 2D numpy array, for further analysis or custom plotting.

#### Simple plots with `vis`
`plot_amr` and `plot_part` project the data along the line of sight and draw it with `imshow`. The map covers the region the data was read from, unless `region` is given.
```python
import matplotlib.pyplot as plt
import pyramis as pr

region = [[0.4, 0.6], [0.4, 0.6], [0.4, 0.6]]
cell = pr.ramses.read_cell(ramses_path, iout=1, region=region)
part = pr.ramses.read_part(ramses_path, iout=1, region=region, part_type='star')

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# mass-weighted mean gas density, projected along z
pr.vis.plot_amr(cell, field='density', weight='mass', mode='mean', projection=['x', 'y'], ax=axes[0])
# stellar surface mass map
pr.vis.plot_part(part, weight='mass', mode='sum', projection=['x', 'y'], shape=512, ax=axes[1])

# add a physical scale bar
pr.vis.add_ruler(cell.info, region=cell.region, ax=axes[0])
plt.show()
```
- `mode` sets how values along the line of sight are combined: `'sum'`, `'mean'`, `'min'` or `'max'`.
- `norm` sets the colour scale: `'auto'` (default: log if every pixel is positive, linear otherwise), `'log'`, `'linear'`, `'asinh'` or `'none'`.
- `cmap` and `kw_imshow` are passed on to `imshow`, and other keyword arguments such as `shape` are passed on to the projection functions below.
- To draw a sub-region, pass a region object, e.g. `region=pr.geometry.Box([[0.45, 0.55], [0.45, 0.55], [0.45, 0.55]])`.

Points and circles can be overlaid on the same axis, for example to mark halos:
```python
pr.vis.plot_points(part, projection=['x', 'y'], s=1, color='white')
pr.vis.plot_circles(halo, radius_name='virial_radius', projection=['x', 'y'], fill=False, color='red')
```

#### Array output with `im`
`amr_projection` and `part_projection` return the projected map as a 2D array instead of drawing it. They take a position array of shape (N, 3) and the per-element quantities directly.
```python
import numpy as np

lims = [[0.4, 0.6], [0.4, 0.6], [0.4, 0.6]]

gas_map = pr.im.amr_projection(
    cell['position'], levels=cell['level'],
    quantities=cell['density'], weights=cell['mass'],
    mode='mean', lims=lims, shape=512, projection=['x', 'y'])

star_map = pr.im.part_projection(
    part['position'], weights=part['mass'],
    mode='sum', lims=lims, shape=512, projection=['x', 'y'],
    plot_method='cic')  # 'hist' (default) or 'cic' (cloud-in-cell)

print(gas_map.shape)  # (512, 512)
plt.imshow(np.log10(gas_map), origin='lower', extent=[0.4, 0.6, 0.4, 0.6])
```
For 2D scatter data, `pr.im.density_2d(x, y, lims, shape=..., weights=..., method=...)` builds a density map with `'hist'`, `'cic'` or `'dtfe'`.
