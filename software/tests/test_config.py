"""Tests for the config module."""

from pathlib import Path

import pytest
from loaf.config import Config, get_config, load_config


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_yaml(self, sample_config_yaml: Path) -> None:
        """Test loading a valid YAML config file."""
        config = load_config(sample_config_yaml)

        assert isinstance(config, dict)
        assert "region" in config
        assert config["region"]["name"] == "test_region"

    def test_load_missing_file(self, tmp_config_dir: Path) -> None:
        """Test that loading a missing file raises FileNotFoundError."""
        missing_path = tmp_config_dir / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            load_config(missing_path)

    def test_load_seattle_config(self, seattle_config_path: Path) -> None:
        """Test loading the actual Seattle config file."""
        if not seattle_config_path.exists():
            pytest.skip("Seattle config not found")

        config = load_config(seattle_config_path)

        assert config["region"]["name"] == "seattle"
        assert config["region"]["lat_min"] == 46.5
        assert config["region"]["lat_max"] == 49.0


class TestConfig:
    """Tests for Config class."""

    def test_attribute_access(self) -> None:
        """Test that Config provides attribute access to dict values."""
        config = Config({"name": "test", "value": 42})

        assert config.name == "test"
        assert config.value == 42

    def test_nested_config(self) -> None:
        """Test that nested dicts become nested Config objects."""
        config = Config({
            "outer": {
                "inner": {
                    "value": "deep"
                }
            }
        })

        assert isinstance(config.outer, Config)
        assert isinstance(config.outer.inner, Config)
        assert config.outer.inner.value == "deep"

    def test_repr(self) -> None:
        """Test Config string representation."""
        config = Config({"name": "test"})
        repr_str = repr(config)

        assert "Config" in repr_str
        assert "name" in repr_str


class TestGetConfig:
    """Tests for get_config function."""

    def test_returns_config_object(self, sample_config_yaml: Path) -> None:
        """Test that get_config returns a Config object."""
        config = get_config(sample_config_yaml)

        assert isinstance(config, Config)
        assert config.region.name == "test_region"
        assert config.data.back_hrs == 24

    def test_seattle_config_structure(self, seattle_config_path: Path) -> None:
        """Test the structure of the Seattle config."""
        if not seattle_config_path.exists():
            pytest.skip("Seattle config not found")

        config = get_config(seattle_config_path)

        # Check all expected sections exist
        assert hasattr(config, "region")
        assert hasattr(config, "data")
        assert hasattr(config, "model")
        assert hasattr(config, "training")

        # Check region values
        assert config.region.lat_min == 46.5
        assert config.region.lon_min == -124.0

        # Check model values
        assert config.model.hidden_dim == 128
        assert config.model.num_heads == 3
