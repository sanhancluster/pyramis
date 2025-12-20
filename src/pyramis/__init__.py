from .config_module import set_config, get_config
from .basic import *

config = get_config()

__all__ = ["config", "set_config"]