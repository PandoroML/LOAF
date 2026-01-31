"""Data download modules for ERA5, HRRR, IEM, and MADIS."""

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
from loaf.data.download.iem import (
    convert_to_model_units,
    download_iem_range,
    download_iem_region,
    download_iem_station,
    download_iem_stations,
    get_available_stations,
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
    # IEM (ASOS/AWOS - no registration required)
    "download_iem_station",
    "download_iem_stations",
    "download_iem_region",
    "download_iem_range",
    "get_available_stations",
    "convert_to_model_units",
]
