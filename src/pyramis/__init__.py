from .config_module import load_config
config = load_config()

from .basic import *

__all__ = ["config"]