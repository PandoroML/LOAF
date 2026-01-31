"""HRRR (High Resolution Rapid Refresh) data download module.

Downloads HRRR forecast data from NOAA via AWS S3 using the Herbie library.
HRRR provides 3km resolution forecasts covering CONUS with hourly updates.

No authentication required - data is freely available on AWS S3.

Reference: https://rapidrefresh.noaa.gov/hrrr/
Herbie docs: https://herbie.readthedocs.io/
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import xarray as xr
from herbie import Herbie
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Default HRRR variables for wind prediction
# Format is GRIB2 search pattern for Herbie
DEFAULT_VARIABLES = "(?:TMP:2 m|DPT:2 m|UGRD:10 m|VGRD:10 m)"

# Seattle/PNW region bounds
SEATTLE_BOUNDS = {
    "lat_min": 46.5,
    "lat_max": 49.0,
    "lon_min": -124.0 + 360,  # HRRR uses 0-360 longitude
    "lon_max": -121.0 + 360,
}


def download_hrrr_hourly(
    run_time: str | datetime,
    var_list: str = DEFAULT_VARIABLES,
    lat_min: float = SEATTLE_BOUNDS["lat_min"],
    lat_max: float = SEATTLE_BOUNDS["lat_max"],
    lon_min: float = SEATTLE_BOUNDS["lon_min"],
    lon_max: float = SEATTLE_BOUNDS["lon_max"],
    max_lead_hr: int = 18,
    verbose: bool = False,
) -> xr.Dataset | None:
    """Download HRRR data for a single model run time.

    Args:
        run_time: Model initialization time (e.g., "2024-01-15 12:00" or datetime).
        var_list: GRIB2 search pattern for variables to download.
        lat_min: Minimum latitude for spatial subsetting.
        lat_max: Maximum latitude for spatial subsetting.
        lon_min: Minimum longitude (0-360 format for HRRR).
        lon_max: Maximum longitude (0-360 format for HRRR).
        max_lead_hr: Maximum forecast lead time in hours.
        verbose: Whether to print download progress.

    Returns:
        xarray Dataset with dimensions (step, latitude, longitude) containing
        the requested variables, or None if download failed.
    """
    if isinstance(run_time, datetime):
        run_time = run_time.strftime("%Y-%m-%d %H:%M")

    datasets = []

    lead_hours = range(max_lead_hr + 1)
    if verbose:
        lead_hours = tqdm(lead_hours, desc=f"Downloading {run_time}")

    for lead_hr in lead_hours:
        try:
            hrrr_file = Herbie(run_time, model="hrrr", product="sfc", fxx=lead_hr, verbose=False)

            if hrrr_file.grib is None:
                logger.warning(f"No GRIB file found for {run_time} fxx={lead_hr}")
                break

            # Download and parse to xarray
            dss = hrrr_file.xarray(search=var_list)
            dss_new = []

            for ds in dss:
                # Keep only essential coordinates
                ds = ds.drop_vars(
                    [
                        coord
                        for coord in ds.coords
                        if coord not in ["latitude", "longitude", "time", "step"]
                    ]
                )

                # Spatial subsetting
                location_mask = (
                    (ds.latitude >= lat_min)
                    & (ds.latitude <= lat_max)
                    & (ds.longitude >= lon_min)
                    & (ds.longitude <= lon_max)
                )
                ds = ds.where(location_mask, drop=True)

                dss_new.append(ds)

            dataset = xr.merge(dss_new, compat="identical")
            datasets.append(dataset)

        except Exception as e:
            logger.error(f"Failed to download {run_time} fxx={lead_hr}: {e}")
            continue

    if not datasets:
        return None

    return xr.concat(datasets, dim="step")


def download_hrrr_daily(
    date: datetime,
    var_list: str = DEFAULT_VARIABLES,
    lat_min: float = SEATTLE_BOUNDS["lat_min"],
    lat_max: float = SEATTLE_BOUNDS["lat_max"],
    lon_min: float = SEATTLE_BOUNDS["lon_min"],
    lon_max: float = SEATTLE_BOUNDS["lon_max"],
    max_lead_hr: int = 18,
    verbose: bool = True,
) -> xr.Dataset | None:
    """Download HRRR data for all 24 model runs in a day.

    HRRR runs every hour, so a full day includes 24 model initializations.

    Args:
        date: Date to download (time component is ignored).
        var_list: GRIB2 search pattern for variables to download.
        lat_min: Minimum latitude for spatial subsetting.
        lat_max: Maximum latitude for spatial subsetting.
        lon_min: Minimum longitude (0-360 format for HRRR).
        lon_max: Maximum longitude (0-360 format for HRRR).
        max_lead_hr: Maximum forecast lead time in hours.
        verbose: Whether to print download progress.

    Returns:
        xarray Dataset with dimensions (time, step, latitude, longitude),
        or None if all downloads failed.
    """
    # Generate all 24 hourly run times for the day
    base_date = datetime(date.year, date.month, date.day)
    run_times = [(base_date + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M") for i in range(24)]

    datasets = []
    run_iterator = tqdm(run_times, desc=f"Downloading {date.date()}") if verbose else run_times

    for run_time in run_iterator:
        dataset = download_hrrr_hourly(
            run_time,
            var_list,
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            max_lead_hr,
            verbose=False,
        )
        if dataset is not None:
            datasets.append(dataset)

    if not datasets:
        return None

    return xr.concat(datasets, dim="time")


def download_hrrr_range(
    start_date: datetime,
    end_date: datetime,
    output_dir: str | Path,
    var_list: str = DEFAULT_VARIABLES,
    lat_min: float = SEATTLE_BOUNDS["lat_min"],
    lat_max: float = SEATTLE_BOUNDS["lat_max"],
    lon_min: float = SEATTLE_BOUNDS["lon_min"],
    lon_max: float = SEATTLE_BOUNDS["lon_max"],
    max_lead_hr: int = 18,
) -> list[Path]:
    """Download HRRR data for a date range, saving one file per day.

    Args:
        start_date: First date to download.
        end_date: Last date to download (inclusive).
        output_dir: Directory to save NetCDF files.
        var_list: GRIB2 search pattern for variables to download.
        lat_min: Minimum latitude for spatial subsetting.
        lat_max: Maximum latitude for spatial subsetting.
        lon_min: Minimum longitude (0-360 format for HRRR).
        lon_max: Maximum longitude (0-360 format for HRRR).
        max_lead_hr: Maximum forecast lead time in hours.

    Returns:
        List of paths to saved NetCDF files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    current_date = start_date

    while current_date <= end_date:
        filename = output_dir / f"hrrr_{current_date.strftime('%Y%m%d')}.nc"

        if filename.exists():
            logger.info(f"Skipping {filename} - already exists")
            saved_files.append(filename)
            current_date += timedelta(days=1)
            continue

        logger.info(f"Downloading HRRR for {current_date.date()}")
        dataset = download_hrrr_daily(
            current_date,
            var_list,
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            max_lead_hr,
        )

        if dataset is not None:
            dataset.to_netcdf(filename)
            logger.info(f"Saved {filename}")
            saved_files.append(filename)
        else:
            logger.warning(f"No data downloaded for {current_date.date()}")

        current_date += timedelta(days=1)

    return saved_files


def main() -> None:
    """CLI entry point for HRRR download."""
    parser = argparse.ArgumentParser(
        description="Download HRRR forecast data for a specified region and date range."
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default="data/hrrr",
        help="Output directory for NetCDF files (default: data/hrrr)",
    )
    parser.add_argument(
        "--var-list",
        default=DEFAULT_VARIABLES,
        help=f"GRIB2 variable search pattern (default: {DEFAULT_VARIABLES})",
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
        default=SEATTLE_BOUNDS["lon_min"] - 360,
        help="Minimum longitude in -180 to 180 format (default: -124.0 for Seattle)",
    )
    parser.add_argument(
        "--lon-max",
        type=float,
        default=SEATTLE_BOUNDS["lon_max"] - 360,
        help="Maximum longitude in -180 to 180 format (default: -121.0 for Seattle)",
    )
    parser.add_argument(
        "--max-lead-hr",
        type=int,
        default=18,
        help="Maximum forecast lead time in hours (default: 18)",
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Single date to download (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for range download (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for range download (YYYY-MM-DD)",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Convert longitude from -180/180 to 0/360 for HRRR
    lon_min = args.lon_min + 360 if args.lon_min < 0 else args.lon_min
    lon_max = args.lon_max + 360 if args.lon_max < 0 else args.lon_max

    if args.date:
        # Single day download
        date = datetime.strptime(args.date, "%Y-%m-%d")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset = download_hrrr_daily(
            date,
            args.var_list,
            args.lat_min,
            args.lat_max,
            lon_min,
            lon_max,
            args.max_lead_hr,
        )

        if dataset is not None:
            filename = output_dir / f"hrrr_{date.strftime('%Y%m%d')}.nc"
            dataset.to_netcdf(filename)
            logger.info(f"Saved {filename}")
        else:
            logger.error(f"Failed to download data for {date.date()}")

    elif args.start_date and args.end_date:
        # Range download
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

        download_hrrr_range(
            start_date,
            end_date,
            args.output_dir,
            args.var_list,
            args.lat_min,
            args.lat_max,
            lon_min,
            lon_max,
            args.max_lead_hr,
        )

    else:
        parser.error("Please specify either --date or both --start-date and --end-date")


if __name__ == "__main__":
    main()
