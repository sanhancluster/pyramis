import tomllib
from pathlib import Path
import multiprocessing as mp

CONFIG_PATH = Path(__file__).parent / "config.toml"
BASE_CONFIG_PATH = Path(__file__).parent / "config_base.toml"

def load_config(path: str | Path = CONFIG_PATH) -> dict:
    """Load configuration from a TOML file."""
    base_path = BASE_CONFIG_PATH
    path = Path(path)

    if not path.exists():
        path = BASE_CONFIG_PATH
    
    with base_path.open("rb") as f:
        config = tomllib.load(f)

    with path.open("rb") as f:
        config.update(tomllib.load(f))
        return config
