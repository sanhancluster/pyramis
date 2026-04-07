# Pyramis: Python Reader for Adaptive Mesh Interface Simulations
A python library, to provide essential and efficient framework for management and analysis of the adaptive mesh simulation such as [RAMSES](https://github.com/ramses-organisation/ramses).

## Installing
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
To create new conda environment specifically for Pyramis,
```bash
git clone https://github.com/sanhancluster/pyramis
cd pyramis

conda env create -f environment.yml
conda activate pyramis

python -m pip install -e .
```
If you want to install in an already existing environment,
```bash
git clone https://github.com/sanhancluster/pyramis
cd pyramis

conda activate <my_env_name>
conda env update -f environment.yml

python -m pip install -e .
```

## How to use
### Reading the RAMSES raw data
pyramis uses multi-threading (concurrent.futures.ThreadPoolExecutor) by default for reading RAMSES snapshot files. Number of workers will be automatically decided by the available resources when the package is imported.

#### The particle data
You can read particle data directly from specific region by following commands. Pyramis computes list of cpu domains to read the complete data within the region using Peano-Hilbert space filling curve.
```python
import pyramis as pr
# path to the directory where output_* are located
ramses_path = '/path/to/ramses/'
# targeting box to read in code unit
region = [[0.4, 0.6], [0.4, 0.6], [0.4, 0.6]]

part = pr.ramses.read_part(ramses_path, iout=1, region=region) # reads output_00001

print(f"Total particle mass within the box is {np.sum(part['m'])}") # in code unit
```
For a particular type of particles, ```part_type``` option can be used.
```python
part = pr.ramses.read_part(ramses_path, iout=1, part_type='star')
```

#### The cell data
You can read all cells from specific region by following commands.
```python
# path to the directory where output_* are located
ramses_path = '/path/to/ramses/'
# targeting box to read in code unit
region = [[0.4, 0.6], [0.4, 0.6], [0.4, 0.6]]

cell = pr.ramses.read_cell(ramses_path, iout=1, region=region)

print(f"Mean gas density within the box is {np.mean(cell['rho'])}") # in code unit
```

### Reading the RAMSES HDF format data
pyramis uses concurrent.futures.ProcessPoolExecutor by default to enable parallel read from HDF files. This requires main block guard for the top-level script.

#### The particle and cell data
```python
if __name__ == '__main__':
    # path to the directory where part_*.h5, cell_*.h5 are located
    hdf_path = '/path/to/hdf/'
    # targeting box in code unit
    region = [[0.4, 0.6], [0.4, 0.6], [0.4, 0.6]]

    part = pr.hdf.read_part(hdf_path, part_type='star', iout=1, region=region)
    cell = pr.hdf.read_cell(hdf_path, iout=1, region=region)
```
Particle type (```part_type```) need to be always present for reading particle data.

### Reading the Dyablo cell data
```python
dyablo_path = '/path/to/dyablo/'

pr.dyablo.read_cell(dyablo_path, istep=0)
```

### Reading the HaloMaker data

#### DM Halo and Galaxy Catalog
```python
halomaker_path = '/path/to/halos/' # path the directory where tree_bricks* are located

halo = pr.halo_finder.read_halomaker(halomaker_path, iout=1)
galaxy = pr.halo_finder.read_halomaker(galaxymaker_path, iout=1, galaxy=True)
```