from __future__ import annotations

import tomllib
from importlib.resources import files
from typing import Any


BASE_CONFIG = "config_base.toml"
_config = {}

def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v

def _load_packaged_toml(name: str) -> dict[str, Any]:
    if not __package__:
        raise RuntimeError("Import this module as part of the 'pyramis' package (do not run it as a script).")

    with files(__package__).joinpath(name).open("rb") as f:
        return tomllib.load(f)

def _load_external_toml(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)

def load_config(path=None) -> dict[str, Any]:
    global _config
    _deep_update(_config, _load_packaged_toml(BASE_CONFIG))
    if path is not None:
        try:
            override = _load_external_toml(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file '{path}' not found.")
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"Error parsing config file '{path}': {e}")
        _deep_update(_config, override)

def get_config():
    global _config
    if not _config:
        load_config()
    return _config

def set_config(key, value):
    cfg = get_config()
    cfg[key] = value

def _init_worker_config(cfg):
    config = get_config()
    _deep_update(config, cfg)