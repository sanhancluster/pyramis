from __future__ import annotations

import tomllib
from importlib.resources import files
from typing import Any


PACKAGE = __package__ or "pyramis"
BASE_CONFIG = "config_base.toml"
CONFIG = "config.toml"


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst


def _load_packaged_toml(name: str) -> dict[str, Any]:
    return tomllib.loads(data)


def load_config() -> dict[str, Any]:
    """
    Load configuration bundled with the package.
    Order:
      1) config_base.toml (required)
      2) config.toml (optional, overrides base)
    """
    config = _load_packaged_toml(BASE_CONFIG)

    try:
        data = files(PACKAGE).joinpath(CONFIG).read_bytes()
        override = tomllib.loads(data)

    except FileNotFoundError:
        return config

    return _deep_update(config, override)