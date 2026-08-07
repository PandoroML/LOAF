"""Iowa Environmental Mesonet (IEM) ASOS/AWOS data download module.

Downloads surface observation data from ASOS/AWOS stations via the Iowa
Environmental Mesonet. No registration required - data is freely available.

This module serves as an alternative to MADIS for initial development and
testing. IEM provides fewer stations than MADIS but has instant access.

Reference: https://mesonet.agron.iastate.edu/
"""

import argparse
import io
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# IEM ASOS download endpoint
IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# Seattle/PNW region bounds (matching other modules)
SEATTLE_BOUNDS = {
    "lat_min": 46.5,
    "lat_max": 49.0,
    "lon_min": -124.0,
    "lon_max": -121.0,
}

# PNW ASOS/AWOS stations within the Seattle region bounds
# These are the major stations - IEM has many more available
PNW_STATIONS = [
    # Washington - Major airports
    "SEA",  # Seattle-Tacoma International
    "BFI",  # Boeing Field
    "PAE",  # Paine Field (Everett)
    "BLI",  # Bellingham International
    "OLM",  # Olympia Regional
    "GRF",  # Gray Army Airfield (Tacoma)
    "RNT",  # Renton Municipal
    "TIW",  # Tacoma Narrows
    "SFF",  # Felts Field (Spokane area, edge of region)
    "CLM",  # Port Angeles
    "FHR",  # Friday Harbor
    "AWO",  # Arlington Municipal
    "S50",  # Auburn Municipal
    "PWT",  # Bremerton National
    "SHN",  # Shelton Sanderson Field
    "0S9",  # Jefferson County International
    # Oregon - Northern stations in region
    "PDX",  # Portland International
    "TTD",  # Troutdale (Portland-Troutdale)
    "HIO",  # Hillsboro
    "UAO",  # Aurora State
    "MMV",  # McMinnville Municipal
    "SLE",  # Salem McNary Field
    # British Columbia border stations (some available via IEM)
    "YVR",  # Vancouver International (if available)
    "YXX",  # Abbotsford (if available)
]

# Default variables to request
# tmpc: Temperature in Celsius
# dwpc: Dewpoint in Celsius
# sknt: Wind speed in knots
# drct: Wind direction in degrees
# p01i: 1-hour precipitation in inches
# alti: Altimeter setting in inches Hg
# mslp: Sea level pressure in millibars
# vsby: Visibility in miles
DEFAULT_VARIABLES = ["tmpc", "dwpc", "sknt", "drct"]


def _bbox_of_geometry(geometry: dict) -> tuple[float, float, float, float]:
    """Compute (lon_min, lon_max, lat_min, lat_max) for a GeoJSON geometry."""
    lons: list[float] = []
    lats: list[float] = []

    def _walk(coords: Any) -> None:
        if isinstance(coords[0], (int, float)):
            lons.append(coords[0])
            lats.append(coords[1])
        else:
            for c in coords:
                _walk(c)

    _walk(geometry["coordinates"])
    return min(lons), max(lons), min(lats), max(lats)


def _find_asos_networks_for_bbox(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> list[str]:
    """Find IEM ASOS network IDs whose coverage overlaps a bounding box.

    IEM does not expose a single global "ASOS" network - stations are split
    into per-country/per-state networks (e.g. "VA_ASOS", "MD_ASOS"). This
    queries IEM's network list and returns the ASOS network IDs whose
    coverage polygon overlaps the given region, so callers can fetch stations
    from just those networks.

    Args:
        lat_min: Minimum latitude.
        lat_max: Maximum latitude.
        lon_min: Minimum longitude.
        lon_max: Maximum longitude.

    Returns:
        List of matching network IDs (e.g. ["VA_ASOS", "MD_ASOS", "DC_ASOS"]).
    """
    url = "https://mesonet.agron.iastate.edu/geojson/networks.geojson"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Failed to fetch network list: {e}")
        return []

    matches = []
    for feature in data.get("features", []):
        network_id = feature.get("id", "")
        geometry = feature.get("geometry")
        if not network_id.endswith("_ASOS") or not geometry:
            continue

        try:
            net_lon_min, net_lon_max, net_lat_min, net_lat_max = _bbox_of_geometry(geometry)
        except (KeyError, IndexError, TypeError):
            continue

        if net_lon_max < lon_min or net_lon_min > lon_max:
            continue
        if net_lat_max < lat_min or net_lat_min > lat_max:
            continue

        matches.append(network_id)

    return matches


def get_available_stations(
    lat_min: float = SEATTLE_BOUNDS["lat_min"],
    lat_max: float = SEATTLE_BOUNDS["lat_max"],
    lon_min: float = SEATTLE_BOUNDS["lon_min"],
    lon_max: float = SEATTLE_BOUNDS["lon_max"],
) -> pd.DataFrame:
    """Fetch list of available ASOS stations within a bounding box.

    Args:
        lat_min: Minimum latitude.
        lat_max: Maximum latitude.
        lon_min: Minimum longitude.
        lon_max: Maximum longitude.

    Returns:
        DataFrame with station metadata (id, name, lat, lon, elevation).
    """
    networks = _find_asos_networks_for_bbox(lat_min, lat_max, lon_min, lon_max)
    if not networks:
        logger.warning("No ASOS networks found covering the requested region")
        return pd.DataFrame()

    stations = []
    for network_id in networks:
        url = f"https://mesonet.agron.iastate.edu/geojson/network/{network_id}.geojson"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch station list for {network_id}: {e}")
            continue

        for feature in data.get("features", []):
            props = feature.get("properties", {})
            coords = feature.get("geometry", {}).get("coordinates", [None, None])

            lon, lat = coords[0], coords[1]
            if lon is None or lat is None:
                continue

            # Filter to bounding box
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                stations.append(
                    {
                        "station_id": props.get("sid", ""),
                        "name": props.get("sname", ""),
                        "lat": lat,
                        "lon": lon,
                        "elevation": props.get("elevation", None),
                    }
                )

    df = pd.DataFrame(stations)
    logger.info(f"Found {len(df)} ASOS stations in region across {len(networks)} network(s)")
    return df


def download_iem_station(
    station: str,
    start_date: datetime,
    end_date: datetime,
    variables: list[str] = DEFAULT_VARIABLES,
    include_latlon: bool = True,
) -> pd.DataFrame | None:
    """Download IEM ASOS data for a single station.

    Args:
        station: ASOS station identifier (e.g., "SEA").
        start_date: Start date for data request.
        end_date: End date for data request (inclusive).
        variables: List of variables to download.
        include_latlon: Whether to include lat/lon in output.

    Returns:
        DataFrame with observation data, or None if download failed.
    """
    params = {
        "station": station,
        "data": ",".join(variables),
        "year1": start_date.year,
        "month1": start_date.month,
        "day1": start_date.day,
        "year2": end_date.year,
        "month2": end_date.month,
        "day2": end_date.day,
        "tz": "UTC",
        "format": "onlycomma",
        "latlon": "yes" if include_latlon else "no",
        "elev": "yes",
        "missing": "null",
        "trace": "null",
        "direct": "no",
        "report_type": "3",  # METAR and special reports
    }

    try:
        response = requests.get(IEM_ASOS_URL, params=params, timeout=60)
        response.raise_for_status()

        # Parse CSV response
        df = pd.read_csv(io.StringIO(response.text))

        if df.empty:
            logger.warning(f"No data returned for station {station}")
            return None

        # Parse timestamp
        if "valid" in df.columns:
            df["valid"] = pd.to_datetime(df["valid"], utc=True)
            df = df.rename(columns={"valid": "time"})

        logger.debug(f"Downloaded {len(df)} observations for {station}")
        return df

    except Exception as e:
        logger.error(f"Failed to download data for {station}: {e}")
        return None


def download_iem_stations(
    stations: list[str],
    start_date: datetime,
    end_date: datetime,
    variables: list[str] = DEFAULT_VARIABLES,
    request_delay: float = 1.0,
) -> pd.DataFrame:
    """Download IEM ASOS data for multiple stations.

    Args:
        stations: List of ASOS station identifiers.
        start_date: Start date for data request.
        end_date: End date for data request (inclusive).
        variables: List of variables to download.
        request_delay: Seconds to wait between station requests, to avoid
            IEM's rate limiting (HTTP 429).

    Returns:
        DataFrame with observation data from all stations.
    """
    all_data = []

    for i, station in enumerate(stations):
        if i > 0 and request_delay > 0:
            time.sleep(request_delay)
        logger.info(f"Downloading {station}...")
        df = download_iem_station(station, start_date, end_date, variables)
        if df is not None and not df.empty:
            all_data.append(df)

    if not all_data:
        logger.warning("No data downloaded from any station")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"Downloaded {len(combined)} total observations from {len(all_data)} stations")
    return combined


def download_iem_region(
    start_date: datetime,
    end_date: datetime,
    lat_min: float = SEATTLE_BOUNDS["lat_min"],
    lat_max: float = SEATTLE_BOUNDS["lat_max"],
    lon_min: float = SEATTLE_BOUNDS["lon_min"],
    lon_max: float = SEATTLE_BOUNDS["lon_max"],
    variables: list[str] = DEFAULT_VARIABLES,
    use_predefined_stations: bool = True,
) -> pd.DataFrame:
    """Download IEM ASOS data for all stations in a region.

    Args:
        start_date: Start date for data request.
        end_date: End date for data request (inclusive).
        lat_min: Minimum latitude.
        lat_max: Maximum latitude.
        lon_min: Minimum longitude.
        lon_max: Maximum longitude.
        variables: List of variables to download.
        use_predefined_stations: If True, use PNW_STATIONS list.
            If False, query IEM for all available stations in region.

    Returns:
        DataFrame with observation data from all stations in region.
    """
    if use_predefined_stations:
        stations = PNW_STATIONS
        logger.info(f"Using {len(stations)} predefined PNW stations")
    else:
        station_df = get_available_stations(lat_min, lat_max, lon_min, lon_max)
        if station_df.empty:
            logger.error("No stations found in region")
            return pd.DataFrame()
        stations = station_df["station_id"].tolist()
        logger.info(f"Found {len(stations)} stations in region")

    return download_iem_stations(stations, start_date, end_date, variables)


def convert_to_model_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert IEM data to units expected by the model.

    Converts:
    - Wind speed: knots -> m/s
    - Wind direction: degrees (unchanged)
    - Temperature: Celsius (unchanged)
    - Dewpoint: Celsius (unchanged)

    Also computes U/V wind components from speed/direction.

    Args:
        df: DataFrame with IEM data.

    Returns:
        DataFrame with converted units and U/V wind components.
    """
    import numpy as np

    df = df.copy()

    # Convert wind speed from knots to m/s
    if "sknt" in df.columns:
        df["wind_speed_ms"] = df["sknt"] * 0.514444

    # Compute U/V components from speed and direction
    # Meteorological convention: direction is where wind comes FROM
    # U = -speed * sin(direction), V = -speed * cos(direction)
    if "sknt" in df.columns and "drct" in df.columns:
        wind_speed_ms = df["sknt"] * 0.514444
        wind_dir_rad = np.radians(df["drct"])
        df["u"] = -wind_speed_ms * np.sin(wind_dir_rad)
        df["v"] = -wind_speed_ms * np.cos(wind_dir_rad)

    # Rename for consistency with ERA5/HRRR loaders
    rename_map = {
        "tmpc": "temp",
        "dwpc": "dewpoint",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    return df


def save_iem_data(
    df: pd.DataFrame,
    output_path: str | Path,
    format: str = "parquet",
) -> Path:
    """Save IEM data to disk.

    Args:
        df: DataFrame with IEM data.
        output_path: Output file path.
        format: Output format ("parquet" or "csv").

    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "parquet":
        df.to_parquet(output_path, index=False)
    elif format == "csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unknown format: {format}")

    logger.info(f"Saved {len(df)} observations to {output_path}")
    return output_path


def download_iem_range(
    start_date: datetime,
    end_date: datetime,
    output_dir: str | Path,
    stations: list[str] | None = None,
    variables: list[str] = DEFAULT_VARIABLES,
    format: str = "parquet",
    request_delay: float = 1.0,
) -> list[Path]:
    """Download IEM data for a date range, saving one file per month.

    Args:
        start_date: First date to download.
        end_date: Last date to download (inclusive).
        output_dir: Directory to save files.
        stations: List of stations to download. If None, uses PNW_STATIONS.
        variables: List of variables to download.
        format: Output format ("parquet" or "csv").
        request_delay: Seconds to wait between station requests, to avoid
            IEM's rate limiting (HTTP 429).

    Returns:
        List of paths to saved files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if stations is None:
        stations = PNW_STATIONS

    saved_files = []

    # Process month by month
    current_year = start_date.year
    current_month = start_date.month

    while (current_year, current_month) <= (end_date.year, end_date.month):
        # Determine month boundaries
        month_start = datetime(current_year, current_month, 1)

        # Find last day of month
        if current_month == 12:
            next_month = datetime(current_year + 1, 1, 1)
        else:
            next_month = datetime(current_year, current_month + 1, 1)
        month_end = next_month - pd.Timedelta(days=1)

        # Clip to requested range
        month_start = max(month_start, start_date)
        month_end = min(month_end, end_date)

        suffix = "parquet" if format == "parquet" else "csv"
        filename = output_dir / f"iem_{current_year}_{current_month:02d}.{suffix}"

        if filename.exists():
            logger.info(f"Skipping {filename} - already exists")
            saved_files.append(filename)
        else:
            logger.info(f"Downloading IEM data for {current_year}-{current_month:02d}")
            df = download_iem_stations(
                stations,
                month_start,
                datetime(month_end.year, month_end.month, month_end.day),
                variables,
                request_delay,
            )

            if not df.empty:
                # Convert units
                df = convert_to_model_units(df)
                save_iem_data(df, filename, format)
                saved_files.append(filename)
            else:
                logger.warning(f"No data for {current_year}-{current_month:02d}")

        # Advance to next month
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1

    return saved_files


def load_iem_settings_from_config(config_path: str) -> dict[str, Any]:
    """Extract IEM-relevant settings from a LOAF config file."""
    from loaf.config import load_config

    cfg = load_config(config_path)
    iem_cfg = cfg.get("data", {}).get("iem", {})
    region_cfg = cfg.get("region", {})
    return {
        "stations": iem_cfg.get("stations"),
        "variables": iem_cfg.get("variables"),
        "output_dir": iem_cfg.get("output_dir"),
        "format": iem_cfg.get("format"),
        "lat_min": region_cfg.get("lat_min"),
        "lat_max": region_cfg.get("lat_max"),
        "lon_min": region_cfg.get("lon_min"),
        "lon_max": region_cfg.get("lon_max"),
    }


def main() -> None:
    """CLI entry point for IEM download."""
    parser = argparse.ArgumentParser(
        description="Download ASOS/AWOS observation data from Iowa Environmental Mesonet."
    )

    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to a LOAF YAML config file (e.g. config/arlington.yaml). "
             "Provides region bounds, station list, and output settings. "
             "CLI flags override config values when both are provided.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Output directory for data files (default: data/iem)",
    )
    parser.add_argument(
        "--stations",
        nargs="+",
        default=None,
        help="Station IDs to download. Overrides config stations list.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Variables to download (default: tmpc,dwpc,sknt,drct)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default=None,
        help="Output format (default: parquet)",
    )
    parser.add_argument(
        "--list-stations",
        action="store_true",
        help="List available stations and exit. Uses region bounds from --config if provided.",
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=1.0,
        help="Seconds to wait between per-station requests, to avoid IEM's "
             "rate limiting (default: 1.0)",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load config defaults, then let CLI args override
    cfg_settings: dict[str, Any] = {}
    if args.config:
        cfg_settings = load_iem_settings_from_config(args.config)
        logger.info(f"Loaded config from {args.config}")

    stations = args.stations or cfg_settings.get("stations")
    variables = args.variables or cfg_settings.get("variables") or DEFAULT_VARIABLES
    output_dir = args.output_dir or cfg_settings.get("output_dir") or "data/iem"
    fmt = args.format or cfg_settings.get("format") or "parquet"

    lat_min = cfg_settings.get("lat_min", SEATTLE_BOUNDS["lat_min"])
    lat_max = cfg_settings.get("lat_max", SEATTLE_BOUNDS["lat_max"])
    lon_min = cfg_settings.get("lon_min", SEATTLE_BOUNDS["lon_min"])
    lon_max = cfg_settings.get("lon_max", SEATTLE_BOUNDS["lon_max"])

    if args.list_stations:
        if stations:
            print("Stations from config/CLI:")
            for s in stations:
                print(f"  {s}")
        print(f"\nQuerying IEM for all available stations in region "
              f"({lat_min}–{lat_max}°N, {lon_min}–{lon_max}°E)...")
        df = get_available_stations(lat_min, lat_max, lon_min, lon_max)
        if not df.empty:
            print(f"\nFound {len(df)} stations:")
            print(df.to_string(index=False))
        return

    # If no stations provided by config or CLI, fall back to bounding-box discovery
    if not stations:
        logger.info("No stations specified — discovering from region bounds")
        station_df = get_available_stations(lat_min, lat_max, lon_min, lon_max)
        if station_df.empty:
            logger.error("No stations found in region — pass --stations or check bounds")
            return
        stations = station_df["station_id"].tolist()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    download_iem_range(
        start_date,
        end_date,
        output_dir,
        stations,
        variables,
        fmt,
        args.rate_limit_delay,
    )


if __name__ == "__main__":
    main()
