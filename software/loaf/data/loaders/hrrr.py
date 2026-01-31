"""HRRR PyTorch data loader.

Loads HRRR forecast data from NetCDF files for training and inference.
Adapted from LocalizedWeather HRRR.py with simplifications for LOAF.

HRRR data has two time dimensions:
- time: Model initialization time (when the forecast was made)
- step: Forecast lead time (hours ahead from initialization)
"""

from pathlib import Path

import numpy as np
import torch
import xarray as xr


class HRRRLoader:
    """Loader for HRRR forecast data.

    HRRR provides high-resolution (3km) forecasts. The data has both
    initialization times and forecast lead times (steps).

    Args:
        data_dir: Directory containing HRRR NetCDF files.
        years: List of years to load, or specific dates.
        lat_bounds: Tuple of (lat_min, lat_max) for spatial subsetting.
        lon_bounds: Tuple of (lon_min, lon_max) for spatial subsetting.
        variables: List of variable names to load.
        reanalysis_only: If True, only use step=0 (analysis, not forecasts).
    """

    # Standard variable renaming from HRRR names to LOAF names
    VARIABLE_RENAME = {
        "t2m": "temp",
        "d2m": "dewpoint",
        "u10": "u",
        "v10": "v",
        "u": "u80",  # 80m winds if available
        "v": "v80",
        "sdswrf": "solar_radiation",
    }

    def __init__(
        self,
        data_dir: str | Path,
        years: list[int] | None = None,
        lat_bounds: tuple[float, float] | None = None,
        lon_bounds: tuple[float, float] | None = None,
        variables: list[str] | None = None,
        reanalysis_only: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.years = years
        self.reanalysis_only = reanalysis_only

        # Default to Seattle region (HRRR uses 0-360 longitude)
        if lat_bounds is None:
            lat_bounds = (46.5, 49.0)
        if lon_bounds is None:
            lon_bounds = (-124.0, -121.0)

        self.lat_min = lat_bounds[0]
        self.lat_max = lat_bounds[1]
        # Convert to 0-360 if needed
        self.lon_min = lon_bounds[0] if lon_bounds[0] >= 0 else lon_bounds[0] + 360
        self.lon_max = lon_bounds[1] if lon_bounds[1] >= 0 else lon_bounds[1] + 360

        # Variables to use
        self.variables = variables or ["u", "v", "temp", "dewpoint"]

        # Lazy loading
        self._data: xr.Dataset | None = None
        self._node_positions: torch.Tensor | None = None

    @property
    def data(self) -> xr.Dataset:
        """Lazy-load the dataset."""
        if self._data is None:
            self._data = self._load_data()
        return self._data

    def _load_data(self) -> xr.Dataset:
        """Load HRRR data from NetCDF files."""
        files = []

        if self.years:
            for year in self.years:
                year_files = sorted(self.data_dir.glob(f"hrrr_{year}*.nc"))
                files.extend(year_files)
        else:
            # Load all available files
            files = sorted(self.data_dir.glob("hrrr_*.nc"))

        if not files:
            raise FileNotFoundError(
                f"No HRRR files found in {self.data_dir}"
            )

        # Load all files
        ds = xr.open_mfdataset(files, combine="by_coords")

        # Apply variable renaming based on what's available
        rename_map = {}
        for old_name, new_name in self.VARIABLE_RENAME.items():
            if old_name in ds.data_vars and new_name not in ds.data_vars:
                rename_map[old_name] = new_name
        if rename_map:
            ds = ds.rename(rename_map)

        return ds

    def load_to_memory(self) -> None:
        """Load all data into memory for faster access."""
        self._data = self.data.load()

    def get_node_positions(self) -> torch.Tensor:
        """Get (lon, lat) positions of all grid nodes.

        Returns:
            Tensor of shape (n_nodes, 2) with [lon, lat] coordinates.
        """
        if self._node_positions is not None:
            return self._node_positions

        # HRRR uses irregular grid, coords are 2D arrays
        if "latitude" in self.data.coords:
            lats = self.data.latitude.values
            lons = self.data.longitude.values
        else:
            # Might be 1D for preprocessed data
            lats = self.data.lat.values
            lons = self.data.lon.values

        # Handle 2D or 1D coordinates
        if lats.ndim == 2:
            lats = lats.flatten()
            lons = lons.flatten()

        # Convert longitude back to -180 to 180 if needed
        lons = np.where(lons > 180, lons - 360, lons)

        self._node_positions = torch.from_numpy(
            np.stack([lons, lats], axis=-1).astype(np.float32)
        )
        return self._node_positions

    def get_sample(
        self,
        time_start: np.datetime64 | str,
        time_end: np.datetime64 | str,
        variables: list[str] | None = None,
        step: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Extract a sample for a time window.

        Args:
            time_start: Start of the time window (initialization times).
            time_end: End of the time window.
            variables: Variables to extract. Defaults to self.variables.
            step: Forecast step/lead time to use. Only used if
                reanalysis_only is False.

        Returns:
            Dictionary mapping variable names to tensors of shape
            (n_nodes, n_times).
        """
        variables = variables or self.variables

        # Select time window
        subset = self.data.sel(time=slice(time_start, time_end))

        # Select step if present
        if "step" in subset.dims:
            if self.reanalysis_only:
                subset = subset.isel(step=0)
            else:
                subset = subset.isel(step=step)

        result = {}
        for var in variables:
            if var in subset.data_vars:
                values = subset[var].values.astype(np.float32)
                # Reshape to (n_nodes, n_times)
                if values.ndim == 3:
                    # (time, y, x) -> (n_nodes, time)
                    n_times = values.shape[0]
                    values = values.reshape(n_times, -1).T
                elif values.ndim == 2:
                    # (time, node) -> (node, time)
                    values = values.T
                result[var] = torch.from_numpy(values)

        return result

    def get_sample_with_forecast(
        self,
        time_start: np.datetime64 | str,
        time_end: np.datetime64 | str,
        n_historical: int,
        n_forecast: int,
        variables: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract sample with both historical analysis and forecast steps.

        This matches the LocalizedWeather approach where historical data
        uses step=0 (analysis) and future data uses forecast steps.

        Args:
            time_start: Start of the time window.
            time_end: End of the time window (includes last historical).
            n_historical: Number of historical time steps to include.
            n_forecast: Number of forecast steps to include.
            variables: Variables to extract.

        Returns:
            Dictionary mapping variable names to tensors of shape
            (n_nodes, n_historical + n_forecast).
        """
        variables = variables or self.variables

        # Select time window
        subset = self.data.sel(time=slice(time_start, time_end))
        times = subset.time.values

        if len(times) < n_historical:
            raise ValueError(
                f"Not enough times in window: {len(times)} < {n_historical}"
            )

        result = {}
        for var in variables:
            if var not in subset.data_vars:
                continue

            # Historical: use step=0 for all times
            historical = subset[var].isel(step=0, time=slice(0, n_historical))

            # Forecast: use increasing steps from last historical time
            last_time_idx = n_historical - 1
            forecast_steps = []
            for step in range(1, n_forecast + 1):
                if step < len(subset.step):
                    forecast_steps.append(
                        subset[var].isel(time=last_time_idx, step=step)
                    )

            if forecast_steps:
                forecast = xr.concat(forecast_steps, dim="step")
                # Combine
                hist_vals = historical.values.astype(np.float32)
                fore_vals = forecast.values.astype(np.float32)

                # Reshape to (n_nodes, time)
                if hist_vals.ndim == 3:
                    n_hist = hist_vals.shape[0]
                    hist_vals = hist_vals.reshape(n_hist, -1).T
                if fore_vals.ndim == 3:
                    n_fore = fore_vals.shape[0]
                    fore_vals = fore_vals.reshape(n_fore, -1).T

                combined = np.concatenate([hist_vals, fore_vals], axis=-1)
                result[var] = torch.from_numpy(combined)
            else:
                # No forecast steps available
                hist_vals = historical.values.astype(np.float32)
                if hist_vals.ndim == 3:
                    n_hist = hist_vals.shape[0]
                    hist_vals = hist_vals.reshape(n_hist, -1).T
                result[var] = torch.from_numpy(hist_vals)

        return result

    @property
    def n_nodes(self) -> int:
        """Total number of grid nodes."""
        # Get shape from first variable
        for var in self.data.data_vars:
            shape = self.data[var].shape
            # Find spatial dimensions (not time or step)
            if "step" in self.data[var].dims and "time" in self.data[var].dims:
                # (time, step, y, x) or (time, step, node)
                if len(shape) == 4:
                    return shape[2] * shape[3]
                return shape[2]
            elif "time" in self.data[var].dims:
                # (time, y, x) or (time, node)
                if len(shape) == 3:
                    return shape[1] * shape[2]
                return shape[1]
        return 0

    @property
    def times(self) -> np.ndarray:
        """Available initialization times."""
        return self.data.time.values

    @property
    def steps(self) -> np.ndarray | None:
        """Available forecast steps (lead times)."""
        if "step" in self.data.dims:
            return self.data.step.values
        return None

    def compute_statistics(
        self,
        variables: list[str] | None = None,
        step: int = 0,
    ) -> dict[str, dict[str, float]]:
        """Compute min, max, mean, std for each variable.

        Args:
            variables: Variables to compute stats for.
            step: Which forecast step to use for statistics.

        Returns:
            Dictionary mapping variable names to dicts with keys
            'min', 'max', 'mean', 'std'.
        """
        variables = variables or list(self.data.data_vars)

        # Select step if present
        subset = self.data
        if "step" in subset.dims:
            subset = subset.isel(step=step)

        stats = {}
        for var in variables:
            if var in subset.data_vars:
                values = subset[var].values
                stats[var] = {
                    "min": float(np.nanmin(values)),
                    "max": float(np.nanmax(values)),
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values)),
                }

        return stats
