"""Data download modules for ERA5, HRRR, and MADIS."""

from loaf.data.download.era5 import (
    download_era5_month,
    download_era5_range,
    download_era5_year,
    load_era5_month,
)
from loaf.data.download.hrrr import (
    download_hrrr_daily,
    download_hrrr_hourly,
    download_hrrr_range,
)

__all__ = [
    # ERA5
    "download_era5_month",
    "download_era5_year",
    "download_era5_range",
    "load_era5_month",
    # HRRR
    "download_hrrr_daily",
    "download_hrrr_hourly",
    "download_hrrr_range",
]
