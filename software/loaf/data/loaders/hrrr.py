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
        self._valid_mask: np.ndarray | None = None

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

        # Load eagerly and combine in memory - avoids open_mfdataset, which
        # requires dask even when chunks=None in newer xarray versions.
        if len(files) == 1:
            ds = xr.open_dataset(files[0])
        else:
            # override: per-file GRIB provenance attrs (remote_grib, local_grib)
            # differ across files and aren't needed after loading.
            ds = xr.combine_by_coords(
                [xr.open_dataset(f) for f in files], combine_attrs="override"
            )

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

    # A cell needs to be valid in at least this fraction of loaded
    # (variable, time, step) slices to count as a real grid node - see
    # _get_valid_mask.
    _VALID_FRACTION_THRESHOLD = 0.95

    def _get_valid_mask(self) -> np.ndarray:
        """Boolean mask (n_nodes,) of grid cells that are part of the region.

        HRRR's native grid is a rotated (Lambert Conformal) projection, so a
        rectangular lat/lon bounding box doesn't align with it: cells inside
        the requested box's (y, x) index rectangle but outside the true
        lat/lon bounds come back as NaN from the download's spatial masking.
        That's a static, structural property of the grid - every cell inside
        vs. outside the true footprint is the same on every single load.

        This used to require a cell to be non-NaN in literally every loaded
        slice (every variable x time x step) to count as valid, which
        conflated that static geometry with "did every one of potentially
        thousands of independent GRIB fetches happen to succeed for this
        exact cell" - a single bad slice from an isolated download hiccup
        (observed in practice: see test_hrrr_download.py) was enough to
        zero out validity for the *entire* region, since one all-NaN slice
        fails that check everywhere. Requiring only a high fraction (not
        100%) of slices to be valid keeps the real structural boundary
        while tolerating a handful of such gaps. Any gap that does fall
        inside the threshold's tolerance - or inside a specific requested
        sample window - still surfaces loudly: see get_sample()'s NaN check
        below, which is what actually guards training from seeing it.
        """
        if self._valid_mask is not None:
            return self._valid_mask

        n_valid = None
        n_total = 0
        for var in self.data.data_vars:
            values = self.data[var].values
            spatial_size = values.shape[-2] * values.shape[-1]
            flat = values.reshape(-1, spatial_size)
            valid_count = (~np.isnan(flat)).sum(axis=0)
            n_valid = valid_count if n_valid is None else (n_valid + valid_count)
            n_total += flat.shape[0]

        if n_valid is None:
            self._valid_mask = np.array([], dtype=bool)
        else:
            self._valid_mask = (n_valid / n_total) >= self._VALID_FRACTION_THRESHOLD
        return self._valid_mask

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

        positions = np.stack([lons, lats], axis=-1).astype(np.float32)
        positions = positions[self._get_valid_mask()]
        self._node_positions = torch.from_numpy(positions)
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
                values = values[self._get_valid_mask()]
                self._raise_if_nan(values, var, time_start, time_end)
                result[var] = torch.from_numpy(values)

        return result

    def _raise_if_nan(
        self,
        values: np.ndarray,
        var: str,
        time_start: np.datetime64 | str,
        time_end: np.datetime64 | str,
    ) -> None:
        """Fail loudly if a requested window still has missing data.

        _get_valid_mask's fraction threshold (see its docstring) means a
        cell can still be "valid" overall while having a stray NaN at one
        particular time/step - if that time/step happens to fall inside
        *this* request's window, letting it through would silently feed
        NaN into model training/inference instead of surfacing the gap.
        """
        if np.isnan(values).any():
            raise ValueError(
                f"HRRR data for {var!r} has missing values within the "
                f"requested window [{time_start}, {time_end}] - likely a "
                f"gap in the downloaded archive for that day. Re-download "
                f"it (download_hrrr_range will overwrite a file whose "
                f"cached step coverage doesn't match --max-lead-hr, but "
                f"won't detect an internal gap like this on its own)."
            )

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
            (n_nodes, n_historical + n_forecast), restricted to the same
            valid (non-NaN) grid nodes as get_sample()/get_node_positions().

        Raises:
            ValueError: If the window doesn't have n_historical times, or
                the archive doesn't have n_forecast steps beyond step=0 at
                the initialization time (n_historical - 1) - e.g. the data
                was downloaded with a lower --max-lead-hr than n_forecast
                needs. Raised rather than silently returning a shorter
                array, since a caller sizes its model's input width from
                n_historical + n_forecast and a silent shortfall would only
                surface later as an opaque shape mismatch.
        """
        variables = variables or self.variables

        # Select time window
        subset = self.data.sel(time=slice(time_start, time_end))
        times = subset.time.values

        if len(times) < n_historical:
            raise ValueError(
                f"Not enough times in window: {len(times)} < {n_historical}"
            )

        n_steps_available = len(subset.step) - 1 if "step" in subset.dims else 0
        if n_forecast > n_steps_available:
            init_time = times[n_historical - 1]
            raise ValueError(
                f"Requested {n_forecast} HRRR forecast step(s) ahead of "
                f"{init_time}, but only {n_steps_available} are archived. "
                f"Re-download HRRR with a higher --max-lead-hr, or reduce "
                f"lead_times so the forecast horizon fits what's downloaded."
            )

        valid_mask = self._get_valid_mask()
        last_time_idx = n_historical - 1

        result = {}
        for var in variables:
            if var not in subset.data_vars:
                continue

            # Historical: step=0 (analysis) for each of the n_historical times.
            historical = subset[var].isel(step=0, time=slice(0, n_historical))
            hist_vals = historical.values.astype(np.float32)
            if hist_vals.ndim == 3:
                hist_vals = hist_vals.reshape(hist_vals.shape[0], -1).T

            if n_forecast == 0:
                hist_masked = hist_vals[valid_mask]
                self._raise_if_nan(hist_masked, var, time_start, time_end)
                result[var] = torch.from_numpy(hist_masked)
                continue

            # Forecast: steps 1..n_forecast issued from the last historical
            # (i.e. "now") initialization - this is the actual future
            # forecast, not more history.
            forecast = subset[var].isel(
                time=last_time_idx, step=slice(1, n_forecast + 1)
            )
            fore_vals = forecast.values.astype(np.float32)
            if fore_vals.ndim == 3:
                fore_vals = fore_vals.reshape(fore_vals.shape[0], -1).T

            combined = np.concatenate([hist_vals, fore_vals], axis=-1)
            combined_masked = combined[valid_mask]
            self._raise_if_nan(combined_masked, var, time_start, time_end)
            result[var] = torch.from_numpy(combined_masked)

        return result

    @property
    def n_nodes(self) -> int:
        """Total number of valid (non-NaN) grid nodes."""
        return int(self._get_valid_mask().sum())

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
