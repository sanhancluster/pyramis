import os
from .config_module import get_config, set_config, load_config
from types import SimpleNamespace

load_config()
config = get_config()
cgs_unit = SimpleNamespace(**config['CGS_UNIT'])
cgs_constants = SimpleNamespace(**config['CGS_CONSTANTS'])

if config['DEFAULT_N_PROCS'] == 'auto':
    config['DEFAULT_N_PROCS'] = len(os.sched_getaffinity(0))


from .basic import *
from . import utils

timer = utils.Timestamp(verbose_level=config['VERBOSE_LEVEL'])

from . import geometry, halo_finder, image, hdf, ramses, astro
timer.message("Initialization complete for Pyramis.", verbose_lim=2)
