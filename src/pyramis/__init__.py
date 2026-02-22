import os
from .config_module import set_config, get_config
from types import SimpleNamespace

config = get_config()
cgs_unit = SimpleNamespace(**config['CGS_UNIT'])
cgs_constants = SimpleNamespace(**config['CGS_CONSTANTS'])

if config['DEFAULT_N_PROCS'] == 'auto':
    config['DEFAULT_N_PROCS'] = len(os.sched_getaffinity(0))

from .basic import *
from . import geometry, image, hdf, ramses, utils, halo
