"""ERA5 reanalysis data download module.

Downloads ERA5 reanalysis data from the Copernicus Climate Data Store (CDS).
ERA5 provides 31km resolution global reanalysis data with hourly temporal resolution.

Requires CDS API registration and ~/.cdsapirc configuration.
See: https://cds.climate.copernicus.eu/how-to-api

Reference: Hersbach et al. (2020). The ERA5 global reanalysis.
"""

import argparse
import logging
from pathlib import Path

import cdsapi
import xarray as xr

logger = logging.getLogger(__name__)

# Default ERA5 variables for wind prediction (matching LocalizedWeather)
DEFAULT_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_net_solar_radiation",
]

# Seattle/PNW region bounds (matching HRRR module)
SEATTLE_BOUNDS = {
    "lat_min": 46.5,
    "lat_max": 49.0,
    "lon_min": -124.0,
    "lon_max": -121.0,
}

# CDS dataset name
DATASET = "reanalysis-era5-single-levels"


def download_era5_month(
    year: int,
    month: int,
    output_path: str | Path,
    variables: list[str] = DEFAULT_VARIABLES,
    lat_min: float = SEATTLE_BOUNDS["lat_min"],
    lat_max: float = SEATTLE_BOUNDS["lat_max"],
    lon_min: float = SEATTLE_BOUNDS["lon_min"],
    lon_max: float = SEATTLE_BOUNDS["lon_max"],
) -> Path | None:
    """Download ERA5 data for a single month.

    Args:
        year: Year to download (e.g., 2024).
        month: Month to download (1-12).
        output_path: Path to save the NetCDF file.
        variables: List of ERA5 variable names to download.
        lat_min: Minimum latitude for spatial subsetting.
        lat_max: Maximum latitude for spatial subsetting.
        lon_min: Minimum longitude (-180 to 180 format).
        lon_max: Maximum longitude (-180 to 180 format).

    Returns:
        Path to the downloaded file, or None if download failed.
    """
    output_path = Path(output_path)

    if output_path.exists():
        logger.info(f"Skipping {output_path} - already exists")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ERA5 area format: [North, West, South, East]
    area = [lat_max, lon_min, lat_min, lon_max]

    request = {
        "product_type": ["reanalysis"],
        "data_format": "netcdf",
        "variable": variables,
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": area,
    }

    logger.info(f"Downloading ERA5 for {year}-{month:02d} to {output_path}")

    try:
        client = cdsapi.Client()
        client.retrieve(DATASET, request).download(output_path)
        logger.info(f"Successfully downloaded {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to download ERA5 for {year}-{month:02d}: {e}")
        return None


def download_era5_year(
    year: int,
    output_dir: str | Path,
    variables: list[str] = DEFAULT_VARIABLES,
    lat_min: float = SEATTLE_BOUNDS["lat_min"],
    lat_max: float = SEATTLE_BOUNDS["lat_max"],
    lon_min: float = SEATTLE_BOUNDS["lon_min"],
    lon_max: float = SEATTLE_BOUNDS["lon_max"],
) -> list[Path]:
    """Download ERA5 data for an entire year, one file per month.

    Args:
        year: Year to download (e.g., 2024).
        output_dir: Directory to save NetCDF files.
        variables: List of ERA5 variable names to download.
        lat_min: Minimum latitude for spatial subsetting.
        lat_max: Maximum latitude for spatial subsetting.
        lon_min: Minimum longitude (-180 to 180 format).
        lon_max: Maximum longitude (-180 to 180 format).

    Returns:
        List of paths to successfully downloaded files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for month in range(1, 13):
        filename = output_dir / f"era5_{year}_{month:02d}.nc"
        result = download_era5_month(
            year,
            month,
            filename,
            variables,
            lat_min,
            lat_max,
            lon_min,
            lon_max,
        )
        if result is not None:
            saved_files.append(result)

    return saved_files


def download_era5_range(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    output_dir: str | Path,
    variables: list[str] = DEFAULT_VARIABLES,
    lat_min: float = SEATTLE_BOUNDS["lat_min"],
    lat_max: float = SEATTLE_BOUNDS["lat_max"],
    lon_min: float = SEATTLE_BOUNDS["lon_min"],
    lon_max: float = SEATTLE_BOUNDS["lon_max"],
) -> list[Path]:
    """Download ERA5 data for a range of months.

    Args:
        start_year: Starting year.
        start_month: Starting month (1-12).
        end_year: Ending year.
        end_month: Ending month (1-12).
        output_dir: Directory to save NetCDF files.
        variables: List of ERA5 variable names to download.
        lat_min: Minimum latitude for spatial subsetting.
        lat_max: Maximum latitude for spatial subsetting.
        lon_min: Minimum longitude (-180 to 180 format).
        lon_max: Maximum longitude (-180 to 180 format).

    Returns:
        List of paths to successfully downloaded files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    current_year = start_year
    current_month = start_month

    while (current_year, current_month) <= (end_year, end_month):
        filename = output_dir / f"era5_{current_year}_{current_month:02d}.nc"
        result = download_era5_month(
            current_year,
            current_month,
            filename,
            variables,
            lat_min,
            lat_max,
            lon_min,
            lon_max,
        )
        if result is not None:
            saved_files.append(result)

        # Advance to next month
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1

    return saved_files


def load_era5_month(file_path: str | Path) -> xr.Dataset:
    """Load an ERA5 monthly NetCDF file.

    Args:
        file_path: Path to the ERA5 NetCDF file.

    Returns:
        xarray Dataset with renamed variables (u, v, temp, dewpoint, solar_radiation).
    """
    ds = xr.open_dataset(file_path)

    # Rename variables to match LocalizedWeather conventions
    rename_map = {
        "u10": "u",
        "v10": "v",
        "t2m": "temp",
        "d2m": "dewpoint",
        "ssr": "solar_radiation",
    }

    # Only rename variables that exist in the dataset
    rename_map = {k: v for k, v in rename_map.items() if k in ds.data_vars}

    if rename_map:
        ds = ds.rename(rename_map)

    return ds


def main() -> None:
    """CLI entry point for ERA5 download."""
    parser = argparse.ArgumentParser(
        description="Download ERA5 reanalysis data for a specified region and time range."
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default="data/era5",
        help="Output directory for NetCDF files (default: data/era5)",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=DEFAULT_VARIABLES,
        help="ERA5 variables to download",
    )
    parser.add_argument(
        "--lat-min",
        type=float,
        default=SEATTLE_BOUNDS["lat_min"],
        help="Minimum latitude (default: 46.5 for Seattle)",
    )
    parser.add_argument(
        "--lat-max",
        type=float,
        default=SEATTLE_BOUNDS["lat_max"],
        help="Maximum latitude (default: 49.0 for Seattle)",
    )
    parser.add_argument(
        "--lon-min",
        type=float,
        default=SEATTLE_BOUNDS["lon_min"],
        help="Minimum longitude (default: -124.0 for Seattle)",
    )
    parser.add_argument(
        "--lon-max",
        type=float,
        default=SEATTLE_BOUNDS["lon_max"],
        help="Maximum longitude (default: -121.0 for Seattle)",
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Single year to download (downloads all 12 months)",
    )
    parser.add_argument(
        "--month",
        type=int,
        help="Single month to download (requires --year)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        help="Start year for range download",
    )
    parser.add_argument(
        "--start-month",
        type=int,
        default=1,
        help="Start month for range download (default: 1)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        help="End year for range download",
    )
    parser.add_argument(
        "--end-month",
        type=int,
        default=12,
        help="End month for range download (default: 12)",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    output_dir = Path(args.output_dir)

    if args.year and args.month:
        # Single month download
        filename = output_dir / f"era5_{args.year}_{args.month:02d}.nc"
        download_era5_month(
            args.year,
            args.month,
            filename,
            args.variables,
            args.lat_min,
            args.lat_max,
            args.lon_min,
            args.lon_max,
        )

    elif args.year:
        # Full year download
        download_era5_year(
            args.year,
            output_dir,
            args.variables,
            args.lat_min,
            args.lat_max,
            args.lon_min,
            args.lon_max,
        )

    elif args.start_year and args.end_year:
        # Range download
        download_era5_range(
            args.start_year,
            args.start_month,
            args.end_year,
            args.end_month,
            output_dir,
            args.variables,
            args.lat_min,
            args.lat_max,
            args.lon_min,
            args.lon_max,
        )

    else:
        parser.error(
            "Please specify --year (with optional --month), or --start-year and --end-year"
        )


if __name__ == "__main__":
    main()
