"""
config.py
---------
Configuration management — loads YAML config files and provides
a simple interface for accessing configuration values.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_config_cache: Dict[str, Dict] = {}


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Results are cached so the file is only read once per path.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file (relative to project root or absolute).

    Returns
    -------
    dict
    """
    abs_path = str(Path(config_path).resolve())

    if abs_path in _config_cache:
        return _config_cache[abs_path]

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    _config_cache[abs_path] = config
    return config


def get_value(config: Dict, *keys: str, default: Any = None) -> Any:
    """
    Safely retrieve a nested value from a config dict.

    Parameters
    ----------
    config : dict
    *keys : str
        Sequence of keys to traverse.
    default : any
        Value to return if the key path doesn't exist.

    Returns
    -------
    The value at the key path, or default.

    Example
    -------
    >>> get_value(cfg, "training", "learning_rate", default=0.001)
    """
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def merge_configs(*configs: Dict) -> Dict:
    """
    Deep-merge multiple config dicts. Later dicts override earlier ones.

    Parameters
    ----------
    *configs : dict

    Returns
    -------
    Merged dict.
    """
    result: Dict = {}
    for cfg in configs:
        _deep_merge(result, cfg)
    return result


def _deep_merge(base: Dict, override: Dict) -> None:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
