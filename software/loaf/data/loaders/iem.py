"""IEM (Iowa Environmental Mesonet) observation data loader.

Loads ASOS/AWOS station observation data from IEM for training and inference.
This serves as a simpler alternative to MADIS for prototyping.

Adapted from LocalizedWeather Madis.py but simplified for IEM data format.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import xarray as xr
from dateutil import rrule

if TYPE_CHECKING:
    from .stations import StationMetadata


class IEMLoader:
    """Loader for IEM ASOS/AWOS observation data.

    Loads station observations from parquet files and provides
    tensor extraction for use in the GNN + ViT model.

    Args:
        data_dir: Directory containing IEM parquet files.
        year: Year to load data for.
        station_metadata: StationMetadata instance defining which stations.
        variables: List of variable names to load.
    """

    # Standard variable names expected by the model
    REQUIRED_VARIABLES = ["u", "v", "temp", "dewpoint"]

    def __init__(
        self,
        data_dir: str | Path,
        year: int,
        station_metadata: StationMetadata | None = None,
        variables: list[str] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.year = year
        self.station_metadata = station_metadata
        self.variables = variables or self.REQUIRED_VARIABLES

        # Generate timeline for the year
        self.timeline = self._generate_timeline(year)

        # Load data
        self._data: xr.Dataset | None = None

    def _generate_timeline(self, year: int) -> pd.DatetimeIndex:
        """Generate hourly timeline for a year (timezone-naive for xarray compat)."""
        start = datetime(year, 1, 1)
        end = datetime(year + 1, 1, 1)
        times = list(rrule.rrule(rrule.HOURLY, dtstart=start, until=end))[:-1]
        return pd.DatetimeIndex(times)  # tz-naive for xarray compatibility

    @property
    def data(self) -> xr.Dataset:
        """Lazy-load the dataset."""
        if self._data is None:
            self._data = self._load_and_process_data()
        return self._data

    def _load_and_process_data(self) -> xr.Dataset:
        """Load IEM data and convert to xarray Dataset."""
        # Find files for the year
        files = sorted(self.data_dir.glob(f"iem_{self.year}_*.parquet"))
        if not files:
            files = sorted(self.data_dir.glob(f"iem_{self.year}_*.csv"))

        if not files:
            raise FileNotFoundError(
                f"No IEM files found for {self.year} in {self.data_dir}"
            )

        # Load all files
        dfs = []
        for f in files:
            if f.suffix == ".parquet":
                dfs.append(pd.read_parquet(f))
            else:
                dfs.append(pd.read_csv(f, parse_dates=["time"]))

        df = pd.concat(dfs, ignore_index=True)

        # Ensure time is datetime
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True)

        # Get station list
        if self.station_metadata is not None:
            stations = self.station_metadata.station_ids
        else:
            station_col = self._find_column(df, ["station", "station_id", "id"])
            stations = df[station_col].unique().tolist()

        # Filter to our stations
        station_col = self._find_column(df, ["station", "station_id", "id"])
        df = df[df[station_col].isin(stations)]

        # Convert to xarray format (stations x time)
        return self._to_xarray(df, stations)

    def _find_column(self, df: pd.DataFrame, candidates: list[str]) -> str:
        """Find a column name from a list of candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        raise ValueError(f"Cannot find column from {candidates} in {df.columns}")

    def _to_xarray(
        self, df: pd.DataFrame, stations: list[str]
    ) -> xr.Dataset:
        """Convert DataFrame to xarray Dataset with (stations, time) dims."""
        station_col = self._find_column(df, ["station", "station_id", "id"])

        # Create station-to-index mapping
        station_to_idx = {s: i for i, s in enumerate(stations)}
        n_stations = len(stations)
        n_times = len(self.timeline)

        # Initialize data arrays
        data_vars = {}
        is_real_vars = {}

        for var in self.variables:
            data_vars[var] = np.full((n_stations, n_times), np.nan, dtype=np.float32)
            is_real_vars[f"{var}_is_real"] = np.zeros(
                (n_stations, n_times), dtype=np.float32
            )

        # Map times to indices
        time_to_idx = {t: i for i, t in enumerate(self.timeline)}

        # Fill in the data
        for _, row in df.iterrows():
            station = row[station_col]
            if station not in station_to_idx:
                continue

            s_idx = station_to_idx[station]

            # Round time to nearest hour and strip timezone for matching
            obs_time = row["time"]
            if hasattr(obs_time, "round"):
                obs_time = obs_time.round("h")
            else:
                obs_time = pd.Timestamp(obs_time).round("h")

            # Strip timezone if present for matching with tz-naive timeline
            if hasattr(obs_time, "tz") and obs_time.tz is not None:
                obs_time = obs_time.tz_localize(None)

            if obs_time not in time_to_idx:
                continue

            t_idx = time_to_idx[obs_time]

            for var in self.variables:
                if var in row and not pd.isna(row[var]):
                    data_vars[var][s_idx, t_idx] = row[var]
                    is_real_vars[f"{var}_is_real"][s_idx, t_idx] = 1.0

        # Fill NaN values using forward/backward fill
        for var in self.variables:
            arr = data_vars[var]
            # Fill along time dimension for each station
            for s in range(n_stations):
                series = pd.Series(arr[s, :])
                filled = series.ffill().bfill()
                if filled.isna().all():
                    # If all NaN, fill with mean of non-NaN values
                    mean_val = np.nanmean(arr)
                    filled = filled.fillna(mean_val if not np.isnan(mean_val) else 0)
                arr[s, :] = filled.values

        # Get station coordinates
        if self.station_metadata is not None:
            lons = self.station_metadata.lons.numpy()
            lats = self.station_metadata.lats.numpy()
        else:
            # Try to get from DataFrame
            lat_col = self._find_column(df, ["lat", "latitude"])
            lon_col = self._find_column(df, ["lon", "longitude"])
            station_locs = df.groupby(station_col).agg(
                {lat_col: "first", lon_col: "first"}
            )
            lons = np.array([station_locs.loc[s, lon_col] for s in stations])
            lats = np.array([station_locs.loc[s, lat_col] for s in stations])

        # Create xarray Dataset
        ds = xr.Dataset(
            data_vars={
                **{var: (["stations", "time"], data_vars[var]) for var in self.variables},
                **{
                    f"{var}_is_real": (["stations", "time"], is_real_vars[f"{var}_is_real"])
                    for var in self.variables
                },
                "lon": (["stations"], lons),
                "lat": (["stations"], lats),
            },
            coords={
                "stations": np.arange(n_stations),
                "time": self.timeline,  # Preserve timezone
            },
        )

        return ds

    def load_to_memory(self) -> None:
        """Load all data into memory."""
        self._data = self.data.load()

    def get_sample(
        self,
        time_start: np.datetime64 | str,
        time_end: np.datetime64 | str,
        variables: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract observation tensors for a time window.

        Args:
            time_start: Start of the time window.
            time_end: End of the time window.
            variables: Variables to extract. Defaults to self.variables.

        Returns:
            Dictionary mapping variable names to tensors of shape
            (n_stations, n_times).
        """
        variables = variables or self.variables

        subset = self.data.sel(time=slice(time_start, time_end))

        result = {}
        for var in variables:
            if var in subset.data_vars:
                values = subset[var].values.astype(np.float32)
                # Replace any remaining NaN values with 0
                values = np.nan_to_num(values, nan=0.0)
                result[var] = torch.from_numpy(values)

                # Also include "is_real" mask if available
                is_real_var = f"{var}_is_real"
                if is_real_var in subset.data_vars:
                    is_real = subset[is_real_var].values.astype(np.float32)
                    result[is_real_var] = torch.from_numpy(is_real)

        return result

    @property
    def n_stations(self) -> int:
        """Number of stations."""
        return len(self.data.stations)

    @property
    def times(self) -> np.ndarray:
        """Available time coordinates."""
        return self.data.time.values

    @property
    def lons(self) -> torch.Tensor:
        """Station longitudes."""
        return torch.from_numpy(self.data.lon.values.astype(np.float32))

    @property
    def lats(self) -> torch.Tensor:
        """Station latitudes."""
        return torch.from_numpy(self.data.lat.values.astype(np.float32))

    def compute_statistics(
        self,
        variables: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Compute min, max, mean, std for each variable.

        Args:
            variables: Variables to compute stats for.

        Returns:
            Dictionary mapping variable names to dicts with keys
            'min', 'max', 'mean', 'std'.
        """
        variables = variables or self.variables

        stats = {}
        for var in variables:
            if var in self.data.data_vars:
                values = self.data[var].values
                # Only use "real" values for statistics
                is_real_var = f"{var}_is_real"
                if is_real_var in self.data.data_vars:
                    mask = self.data[is_real_var].values > 0
                    values = values[mask]

                if len(values) > 0:
                    stats[var] = {
                        "min": float(np.nanmin(values)),
                        "max": float(np.nanmax(values)),
                        "mean": float(np.nanmean(values)),
                        "std": float(np.nanstd(values)),
                    }
                else:
                    stats[var] = {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.25}

        return stats


class MADISLoader(IEMLoader):
    """Loader for MADIS station observation data.

    This is an alias/extension of IEMLoader for when MADIS data becomes
    available. MADIS has additional quality control flags that can be used.

    The main differences from IEM:
    - More stations (100+ vs ~30-50)
    - Quality control flags (windSpeedDD, windDirDD, temperatureDD)
    - NetCDF format instead of CSV/parquet
    """

    def __init__(
        self,
        data_dir: str | Path,
        year: int,
        station_metadata: StationMetadata | None = None,
        variables: list[str] | None = None,
        qc_flags: list[str] | None = None,
    ):
        self.qc_flags = qc_flags or ["S", "V"]  # Standard and Verified
        super().__init__(data_dir, year, station_metadata, variables)

    # TODO: Implement MADIS-specific loading when data is available
    # This will need to handle NetCDF format and QC flags
