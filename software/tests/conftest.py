"""Pytest configuration and shared fixtures."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# A handful of real DMV-area ASOS/AWOS stations, matching config/arlington.yaml.
# Shared by tests that need a small, real-coordinate "trained region" to exercise
# training/inference against without a network call.
ARLINGTON_STATIONS = {
    "DCA": (38.8512, -77.0402),
    "IAD": (38.9531, -77.4565),
    "HEF": (38.7217, -77.5158),
    "MRB": (39.4009, -77.9847),
    "BWI": (39.1774, -76.6684),
}
REAGAN_NATIONAL = (38.8512, -77.0402)  # DCA itself - a real Arlington, VA coordinate


def make_synthetic_iem_data(iem_dir: Path, year: int, n_hours: int = 200) -> None:
    """Write one IEM-format parquet file of synthetic hourly observations."""
    iem_dir.mkdir(parents=True, exist_ok=True)
    start = datetime(year, 1, 1)
    rng = np.random.default_rng(0)

    rows = []
    for station, (lat, lon) in ARLINGTON_STATIONS.items():
        base_u, base_v = rng.uniform(-5, 5, size=2)
        for h in range(n_hours):
            rows.append(
                {
                    "station": station,
                    "lat": lat,
                    "lon": lon,
                    "time": start + timedelta(hours=h),
                    "u": base_u + np.sin(h / 12) + rng.normal(scale=0.1),
                    "v": base_v + np.cos(h / 12) + rng.normal(scale=0.1),
                    "temp": 15 + 5 * np.sin(h / 24) + rng.normal(scale=0.5),
                    "dewpoint": 8 + 4 * np.sin(h / 24) + rng.normal(scale=0.5),
                }
            )

    pd.DataFrame(rows).to_parquet(iem_dir / f"iem_{year}_01.parquet")


@pytest.fixture
def synthetic_arlington_data(tmp_path: Path) -> tuple[Path, int]:
    """A tmp data/ dir with synthetic Arlington-area IEM data. Returns (data_dir, year)."""
    year = 2024
    data_dir = tmp_path / "data"
    make_synthetic_iem_data(data_dir / "iem", year)
    return data_dir, year


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
