"""Configuration management for LOAF.

Loads YAML configuration files for region settings and model hyperparameters.
"""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the configuration.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


class Config:
    """Configuration container with attribute access."""

    def __init__(self, config_dict: dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:
        return f"Config({vars(self)})"


def get_config(config_path: str | Path) -> Config:
    """Load a YAML configuration file and return as Config object.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Config object with attribute access to configuration values.
    """
    config_dict = load_config(config_path)
    return Config(config_dict)
