from __future__ import annotations

import tomllib
from importlib.resources import files
from typing import Any
import os


BASE_CONFIG = "config_base.toml"
_config = {}

def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
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

def load_config(path=None) -> None:
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
    _resolve_special_values(_config)

def get_config():
    global _config
    if not _config:
        load_config()
    return _config

def set_config(key, value):
    cfg = get_config()
    cfg[key] = value
    _resolve_special_values(cfg)

def _resolve_special_values(cfg):
    if cfg['DEFAULT_N_PROCS'] == 'auto':
        cfg['DEFAULT_N_PROCS'] = len(os.sched_getaffinity(0))

def _init_worker_config(cfg):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"      # OpenMP
    os.environ["MKL_NUM_THREADS"] = "1"      # MKL (Intel)
    os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NumExpr
    global _config
    _config = {}
    _deep_update(_config, cfg)
    _resolve_special_values(_config)


def get_vname(vname: str, name_set: str | None=None):
    config = get_config()
    if name_set is not None:
        mapping = config['VNAME_MAPPING'][name_set]
    else:
        mapping = config['VNAME_MAPPING'][config['VNAME_SET']]
    vname = mapping.get(vname, vname)
    return vname


def get_mapping(name_set_from, name_set_to):
    config = get_config()
    mapping_to = config['VNAME_MAPPING'][name_set_to]
    if name_set_from == 'native':
        return mapping_to

    mapping_from = config['VNAME_MAPPING'][name_set_from]
    # Create reverse mapping from name_set_from
    reverse_from = {v: k for k, v in mapping_from.items() if isinstance(v, str)}

    # Create mapping from name_set_from to name_set_to
    mapping = {}
    for k, v in mapping_from.items():
        if isinstance(v, str):
            mapping[v] = mapping_to.get(k, k)
        else:
            mapping[k] = mapping_to.get(k, v)
    return mapping

