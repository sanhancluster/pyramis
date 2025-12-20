from __future__ import annotations

import tomllib
from importlib.resources import files
from typing import Any


BASE_CONFIG = "config_base.toml"
CONFIG = "config.toml"
_config = None

def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst


def _load_packaged_toml(name: str) -> dict[str, Any]:
    if not __package__:
        raise RuntimeError("Import this module as part of the 'pyramis' package (do not run it as a script).")

    with files(__package__).joinpath(name).open("rb") as f:
        return tomllib.load(f)

def load_config() -> dict[str, Any]:
    config = _load_packaged_toml(BASE_CONFIG)
    try:
        override = _load_packaged_toml(CONFIG)
    except FileNotFoundError:
        return config
    return _deep_update(config, override)

def get_config():
    global _config
    if _config is None:
        _config = load_config()
    return _config

def set_config(key, value):
    cfg = get_config()
    cfg[key] = value
    return cfg