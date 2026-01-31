"""Pytest configuration and shared fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def tmp_config_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for config files."""
    return tmp_path


@pytest.fixture
def sample_config_yaml(tmp_config_dir: Path) -> Path:
    """Create a sample YAML config file for testing."""
    config_content = """
region:
  name: test_region
  lat_min: 46.5
  lat_max: 49.0
  lon_min: -124.0
  lon_max: -121.0

data:
  back_hrs: 24
  lead_hrs: 48

model:
  hidden_dim: 128
  num_layers: 3
"""
    config_path = tmp_config_dir / "test_config.yaml"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def seattle_config_path() -> Path:
    """Path to the actual Seattle config file."""
    return Path(__file__).parent.parent / "config" / "seattle.yaml"
