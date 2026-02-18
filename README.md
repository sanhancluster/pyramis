# PYthon-based Ramses Analyzer MInimaliSt
A minimalist version of [Ramses Univsersal Reader](https://github.com/sanhancluster/rur.git), to provide key essential features for management and analysis of the [RAMSES](https://github.com/ramses-organisation/ramses) simulation data.

## Installing
### Using pip
```bash
pip install pyramis
```
### Using developement mode
```bash
git clone https://github.com/sanhancluster/pyramis
cd pyramis
pip install -e .
```

## How to use
### Reading RAMSES raw data
pyramis uses multi-threading (concurrent.futures.ThreadPoolExecutor) by default for reading RAMSES simulation data. Number of workers will be automatically decided by the available resources when the package is imported.
#### The particle data
You can read particle data directly from specific region by following commands.
```python
from pyramis import io
ramses_path = '/path/to/ramses' # path to the directory where output_* are located
iout = 3 # output number
region = [[0.4, 0.6], [0.4, 0.6], [0.4, 0.6]] # targeting box
part = io.read_part(ramses_path, iout=iout, region=region)
print(f"Total particle mass within the box is {np.sum(part['m'])}") # in code unit
```
For a particular type of particles, ```part_type``` option can be used
```python
part = io.read_part(ramses_path, iout=iout, part_type='star')
```
#### The cell data
You can read all cells from specific region by following commands.
```python
from pyramis import io
ramses_path = '/path/to/ramses' # path to the directory where output_* are located
iout = 3 # output number
region = [[0.4, 0.6], [0.4, 0.6], [0.4, 0.6]] # targeting box
cell = io.read_cell(ramses_path, iout=iout, region=region)
print(f"Mean gas density within the box is {np.mean(cell['rho'])}") # in code unit
```