"""ERA5 PyTorch data loader.

Loads ERA5 reanalysis data from NetCDF files for training and inference.
Adapted from LocalizedWeather ERA5.py with simplifications for LOAF.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import xarray as xr


class ERA5Loader:
    """Loader for ERA5 reanalysis data.

    Loads ERA5 data from NetCDF files and provides tensor extraction
    for use in the GNN + ViT model.

    Args:
        data_dir: Directory containing ERA5 NetCDF files.
        years: List of years to load (e.g., [2024]).
        lat_bounds: Tuple of (lat_min, lat_max) for spatial subsetting.
        lon_bounds: Tuple of (lon_min, lon_max) for spatial subsetting.
        variables: List of variable names to load.
        buffer: Spatial buffer in degrees around the region bounds.
    """

    # Standard variable renaming from ERA5 names to LOAF names
    VARIABLE_RENAME = {
        "u10": "u",
        "v10": "v",
        "t2m": "temp",
        "d2m": "dewpoint",
        "ssr": "solar_radiation",
    }

    def __init__(
        self,
        data_dir: str | Path,
        years: list[int],
        lat_bounds: tuple[float, float] | None = None,
        lon_bounds: tuple[float, float] | None = None,
        variables: list[str] | None = None,
        buffer: float = 1.5,
    ):
        self.data_dir = Path(data_dir)
        self.years = years
        self.buffer = buffer

        # Default to Seattle region
        if lat_bounds is None:
            lat_bounds = (46.5, 49.0)
        if lon_bounds is None:
            lon_bounds = (-124.0, -121.0)

        self.lat_min = lat_bounds[0] - buffer
        self.lat_max = lat_bounds[1] + buffer
        self.lon_min = lon_bounds[0] - buffer
        self.lon_max = lon_bounds[1] + buffer

        # Variables to use
        self.variables = variables or ["u", "v", "temp", "dewpoint"]

        # Load data
        self._data: xr.Dataset | None = None
        self._node_positions: torch.Tensor | None = None

    @property
    def data(self) -> xr.Dataset:
        """Lazy-load the dataset."""
        if self._data is None:
            self._data = self._load_data()
        return self._data

    def _load_data(self) -> xr.Dataset:
        """Load ERA5 data from NetCDF files."""
        file_pattern = "era5_*.nc"
        files = []

        for year in self.years:
            year_files = sorted(self.data_dir.glob(f"era5_{year}_*.nc"))
            files.extend(year_files)

        if not files:
            raise FileNotFoundError(
                f"No ERA5 files found in {self.data_dir} for years {self.years}"
            )

        # Load all files (chunks=None to avoid requiring dask)
        if len(files) == 1:
            ds = xr.open_dataset(files[0])
        else:
            ds = xr.open_mfdataset(files, combine="by_coords", chunks=None)

        # Rename variables to standard names
        rename_map = {k: v for k, v in self.VARIABLE_RENAME.items() if k in ds.data_vars}
        if rename_map:
            ds = ds.rename(rename_map)

        # Handle coordinate naming variations
        if "valid_time" in ds.coords:
            ds = ds.rename({"valid_time": "time"})

        # Subset to region (with buffer)
        # Note: ERA5 latitude is typically in descending order
        if "latitude" in ds.coords:
            lat_coord = "latitude"
            lon_coord = "longitude"
        elif "lat" in ds.coords:
            lat_coord = "lat"
            lon_coord = "lon"
        else:
            raise ValueError(f"Cannot find latitude coordinate in dataset: {list(ds.coords)}")

        # Check if latitude is ascending or descending
        lat_values = ds[lat_coord].values
        if lat_values[0] > lat_values[-1]:
            # Descending latitude
            ds = ds.sel(
                {lat_coord: slice(self.lat_max, self.lat_min)},
            )
        else:
            # Ascending latitude
            ds = ds.sel(
                {lat_coord: slice(self.lat_min, self.lat_max)},
            )

        ds = ds.sel({lon_coord: slice(self.lon_min, self.lon_max)})

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

        # Get coordinate names
        if "latitude" in self.data.coords:
            lat_coord, lon_coord = "latitude", "longitude"
        else:
            lat_coord, lon_coord = "lat", "lon"

        # Stack to node format
        stacked = self.data.stack(node=(lon_coord, lat_coord))
        lons = stacked[lon_coord].values
        lats = stacked[lat_coord].values

        self._node_positions = torch.from_numpy(
            np.stack([lons, lats], axis=-1).astype(np.float32)
        )
        return self._node_positions

    def get_sample(
        self,
        time_start: np.datetime64 | str,
        time_end: np.datetime64 | str,
        variables: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract a sample for a time window.

        Args:
            time_start: Start of the time window.
            time_end: End of the time window.
            variables: Variables to extract. Defaults to self.variables.

        Returns:
            Dictionary mapping variable names to tensors of shape
            (n_nodes, n_times).
        """
        variables = variables or self.variables

        # Select time window
        subset = self.data.sel(time=slice(time_start, time_end))

        # Get coordinate names
        if "latitude" in self.data.coords:
            lat_coord, lon_coord = "latitude", "longitude"
        else:
            lat_coord, lon_coord = "lat", "lon"

        # Stack to node format: (time, lat, lon) -> (time, node)
        stacked = subset.stack(node=(lon_coord, lat_coord))

        result = {}
        for var in variables:
            if var in stacked.data_vars:
                # Shape: (time, node) -> (node, time) for model input
                values = stacked[var].values.astype(np.float32)
                values = np.moveaxis(values, 0, -1)  # (time, node) -> (node, time)
                result[var] = torch.from_numpy(values)

        return result

    def get_grid_sample(
        self,
        time_start: np.datetime64 | str,
        time_end: np.datetime64 | str,
        variables: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract a sample preserving grid structure.

        Args:
            time_start: Start of the time window.
            time_end: End of the time window.
            variables: Variables to extract. Defaults to self.variables.

        Returns:
            Dictionary mapping variable names to tensors of shape
            (time, lat, lon).
        """
        variables = variables or self.variables

        # Select time window
        subset = self.data.sel(time=slice(time_start, time_end))

        result = {}
        for var in variables:
            if var in subset.data_vars:
                values = subset[var].values.astype(np.float32)
                result[var] = torch.from_numpy(values)

        return result

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Return (n_lat, n_lon) shape of the grid."""
        if "latitude" in self.data.coords:
            return (len(self.data.latitude), len(self.data.longitude))
        return (len(self.data.lat), len(self.data.lon))

    @property
    def n_nodes(self) -> int:
        """Total number of grid nodes."""
        shape = self.grid_shape
        return shape[0] * shape[1]

    @property
    def times(self) -> np.ndarray:
        """Available time coordinates."""
        return self.data.time.values

    def compute_statistics(
        self,
        variables: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Compute min, max, mean, std for each variable.

        Args:
            variables: Variables to compute stats for. Defaults to all.

        Returns:
            Dictionary mapping variable names to dicts with keys
            'min', 'max', 'mean', 'std'.
        """
        variables = variables or list(self.data.data_vars)

        stats = {}
        for var in variables:
            if var in self.data.data_vars:
                values = self.data[var].values
                stats[var] = {
                    "min": float(np.nanmin(values)),
                    "max": float(np.nanmax(values)),
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values)),
                }

        return stats
