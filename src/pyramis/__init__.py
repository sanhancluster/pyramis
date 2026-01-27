from .config_module import set_config, get_config
from types import SimpleNamespace

config = get_config()
cgs_unit = SimpleNamespace(**config['CGS_UNIT'])
cgs_constants = SimpleNamespace(**config['CGS_CONSTANTS'])

from .basic import *
from . import geometry, image, io, hdf, utils