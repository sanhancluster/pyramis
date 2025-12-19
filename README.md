# PYthon-based Ramses Analyzer MInimaliSt
A minimalist version of [Ramses Univsersal Reader](https://github.com/sanhancluster/rur.git), to provide key essential features for management and analysis of the [RAMSES](https://github.com/ramses-organisation/ramses) simulation data.

## Installing
### Using pip
```bash
pip install pyramis
```

## How to use
### Reading the particle data
```python
from pyramis import io
ramses_path = '/path/to/ramses' # path to the directory where output_* are located
iout = 3 # output number
region = [[0.4, 0.6], [0.4, 0.6], [0.4, 0.6]] # targeting box
part = io.read_part(ramses_path, iout, region)
print(f"Total particle mass within the box is {np.sum(part['m'])}") # in code unit
```
### Reading the cell data
```python
from pyramis import io
ramses_path = '/path/to/ramses' # path to the directory where output_* are located
iout = 3 # output number
region = [[0.4, 0.6], [0.4, 0.6], [0.4, 0.6]] # targeting box
cell = io.read_cell(ramses_path, iout, region)
print(f"Mean gas density within the box is {np.mean(cell['rho'])}") # in code unit
```
