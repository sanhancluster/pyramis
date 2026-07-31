import os
from .config_module import get_config, set_config, load_config
from platformdirs import user_config_dir

config_path = os.path.join(user_config_dir("pyramis"), "config.toml")
if os.path.exists(config_path):
    load_config(config_path)
config = get_config()
cgs_unit = dict(**config['CGS_UNITS'])
cgs_constants = dict(**config['CGS_CONSTANTS'])


from .basic import *
from . import utils

timer = utils.Timestamp(verbose_level=config['VERBOSE_LEVEL'])

from . import geometry, halo_finder, image, hdf, ramses, dyablo, astro, visualize, image as im, visualize as vis
timer.message("Initialization complete for Pyramis.", verbose_lim=2)
