"""Combined weather dataset for training.

Combines ERA5/HRRR grid data with IEM/MADIS station observations
for the GNN + ViT model.

Adapted from LocalizedWeather MixData.py.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from dateutil import rrule
from torch.utils.data import Dataset

from .era5 import ERA5Loader
from .hrrr import HRRRLoader
from .iem import IEMLoader
from .stations import StationMetadata


class WeatherDataset(Dataset):
    """Combined dataset for weather forecasting.

    Combines gridded data (ERA5 or HRRR) with station observations
    for training the GNN + ViT model.

    Args:
        year: Year of data to load.
        back_hrs: Number of historical hours for input.
        lead_hours: Number of forecast hours (prediction horizon).
        station_metadata: StationMetadata instance.
        station_loader: IEMLoader or MADISLoader instance.
        grid_loader: ERA5Loader or HRRRLoader instance (optional).
        station_vars: List of station variables to use.
        grid_vars: List of grid variables to use.
        normalize: Whether to apply normalization.
    """

    def __init__(
        self,
        year: int,
        back_hrs: int,
        lead_hours: int,
        station_metadata: StationMetadata,
        station_loader: IEMLoader,
        grid_loader: ERA5Loader | HRRRLoader | None = None,
        station_vars: list[str] | None = None,
        grid_vars: list[str] | None = None,
        normalize: bool = True,
    ):
        self.year = year
        self.back_hrs = back_hrs
        self.lead_hours = lead_hours

        self.station_metadata = station_metadata
        self.station_loader = station_loader
        self.grid_loader = grid_loader

        # Default variables
        self.station_vars = station_vars or ["u", "v", "temp", "dewpoint"]
        self.grid_vars = grid_vars or ["u", "v", "temp", "dewpoint"]

        self.normalize = normalize

        # Generate timeline for the year
        self.timeline = self._generate_timeline(year)

        # Compute statistics for normalization
        if normalize:
            self._compute_statistics()
        else:
            self.station_stats = {}
            self.grid_stats = {}

        # Compute coordinate normalizers
        self._setup_coord_normalizers()

    def _generate_timeline(self, year: int) -> pd.DatetimeIndex:
        """Generate hourly timeline for a year."""
        start = datetime(year, 1, 1)
        end = datetime(year + 1, 1, 1)
        times = list(rrule.rrule(rrule.HOURLY, dtstart=start, until=end))[:-1]
        return pd.DatetimeIndex(times, tz="UTC")

    def _compute_statistics(self) -> None:
        """Compute normalization statistics."""
        self.station_stats = self.station_loader.compute_statistics(self.station_vars)

        if self.grid_loader is not None:
            self.grid_stats = self.grid_loader.compute_statistics(self.grid_vars)
        else:
            self.grid_stats = {}

    def _setup_coord_normalizers(self) -> None:
        """Set up coordinate normalization."""
        # Use station bounds for coordinate normalization
        self.lat_min = self.station_metadata.lat_min
        self.lat_max = self.station_metadata.lat_max
        self.lon_min = self.station_metadata.lon_min
        self.lon_max = self.station_metadata.lon_max

    def _normalize_coords(
        self, lons: torch.Tensor, lats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize coordinates to [0, 1]."""
        eps = 1e-5
        norm_lons = (lons - self.lon_min) / (self.lon_max - self.lon_min + eps)
        norm_lats = (lats - self.lat_min) / (self.lat_max - self.lat_min + eps)
        return norm_lons, norm_lats

    def _normalize_var(
        self, values: torch.Tensor, var: str, source: str = "station"
    ) -> torch.Tensor:
        """Normalize a variable using precomputed statistics."""
        stats = self.station_stats if source == "station" else self.grid_stats

        if var not in stats:
            return values

        min_val = stats[var]["min"]
        max_val = stats[var]["max"]
        eps = 1e-5

        return (values - min_val) / (max_val - min_val + eps)

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        # We need back_hrs of history and lead_hours of future
        return len(self.timeline) - self.back_hrs - self.lead_hours

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a training sample.

        Args:
            index: Sample index.

        Returns:
            Dictionary containing:
            - time: Tensor of timestamps
            - station_lon, station_lat: Normalized station coordinates
            - k_edge_index: Station graph edges
            - {var}: Station observations for each variable
            - {var}_is_real: Mask of real vs filled values
            - ext_{var}: Grid data for each variable (if grid_loader provided)
            - grid_lon, grid_lat: Normalized grid coordinates
            - ex2m_edge_index: Grid-to-station edges
        """
        # Time window
        start_idx = index
        end_idx = index + self.back_hrs + self.lead_hours

        time_sel = self.timeline[start_idx : end_idx + 1]
        time_start = time_sel[0]
        time_end = time_sel[-1]

        # Build sample dictionary
        sample = {}

        # Timestamps
        sample["time"] = torch.tensor(
            [t.value for t in time_sel], dtype=torch.long
        )

        # Station coordinates (normalized)
        station_lons = self.station_metadata.lons
        station_lats = self.station_metadata.lats
        norm_lons, norm_lats = self._normalize_coords(station_lons, station_lats)
        sample["station_lon"] = norm_lons
        sample["station_lat"] = norm_lats

        # Station graph
        sample["k_edge_index"] = self.station_metadata.get_k_edge_index()

        # Station observations
        station_data = self.station_loader.get_sample(
            time_start, time_end, self.station_vars
        )

        for var in self.station_vars:
            if var in station_data:
                values = station_data[var]
                if self.normalize:
                    values = self._normalize_var(values, var, "station")
                sample[var] = values

            # Include is_real mask
            is_real_var = f"{var}_is_real"
            if is_real_var in station_data:
                sample[is_real_var] = station_data[is_real_var]

        # Grid data (if available)
        if self.grid_loader is not None:
            grid_data = self.grid_loader.get_sample(
                time_start, time_end, self.grid_vars
            )

            for var in self.grid_vars:
                if var in grid_data:
                    values = grid_data[var]
                    if self.normalize:
                        values = self._normalize_var(values, var, "grid")
                    sample[f"ext_{var}"] = values

            # Grid coordinates
            grid_pos = self.grid_loader.get_node_positions()
            grid_lons = grid_pos[:, 0]
            grid_lats = grid_pos[:, 1]
            norm_grid_lons, norm_grid_lats = self._normalize_coords(
                grid_lons, grid_lats
            )
            sample["grid_lon"] = norm_grid_lons
            sample["grid_lat"] = norm_grid_lats

            # Grid-to-station edges (can be precomputed)
            # For now, use simple KNN from grid to stations
            sample["ex2m_edge_index"] = self._compute_grid_to_station_edges(
                grid_pos, self.station_metadata.positions
            )

        return sample

    def _compute_grid_to_station_edges(
        self,
        grid_pos: torch.Tensor,
        station_pos: torch.Tensor,
        k: int = 4,
    ) -> torch.Tensor:
        """Compute edges from grid nodes to nearby stations.

        Args:
            grid_pos: Grid node positions (n_grid, 2).
            station_pos: Station positions (n_stations, 2).
            k: Number of nearest grid nodes per station.

        Returns:
            Edge index of shape (2, n_edges) where edges go from
            grid nodes to stations.
        """
        # For each station, find k nearest grid nodes
        n_stations = station_pos.shape[0]

        # Compute pairwise distances
        diff = station_pos.unsqueeze(1) - grid_pos.unsqueeze(0)
        distances = torch.sqrt(torch.sum(diff ** 2, dim=-1))

        # Find k nearest grid nodes for each station
        _, indices = torch.topk(distances, k, dim=1, largest=False)

        # Build edge index (grid -> station)
        edge_src = indices.flatten()  # Grid node indices
        edge_dst = torch.arange(n_stations).repeat_interleave(k)  # Station indices

        return torch.stack([edge_src, edge_dst], dim=0)

    @property
    def n_stations(self) -> int:
        """Number of stations."""
        return self.station_metadata.n_stations

    @property
    def n_grid_nodes(self) -> int | None:
        """Number of grid nodes (if grid data available)."""
        if self.grid_loader is not None:
            return self.grid_loader.n_nodes
        return None


def create_dataloaders(
    data_dir: str | Path,
    year: int,
    back_hrs: int = 24,
    lead_hours: int = 48,
    batch_size: int = 32,
    val_split: float = 0.15,
    num_workers: int = 4,
    lat_bounds: tuple[float, float] | None = None,
    lon_bounds: tuple[float, float] | None = None,
    use_era5: bool = False,
    use_hrrr: bool = True,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders.

    Args:
        data_dir: Base directory containing data subdirectories.
        year: Year of data to use.
        back_hrs: Number of historical hours.
        lead_hours: Forecast horizon in hours.
        batch_size: Batch size for training.
        val_split: Fraction of data for validation.
        num_workers: Number of dataloader workers.
        lat_bounds: Geographic bounds (lat_min, lat_max).
        lon_bounds: Geographic bounds (lon_min, lon_max).
        use_era5: Whether to use ERA5 as grid data.
        use_hrrr: Whether to use HRRR as grid data.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    data_dir = Path(data_dir)

    # Default Seattle bounds
    if lat_bounds is None:
        lat_bounds = (46.5, 49.0)
    if lon_bounds is None:
        lon_bounds = (-124.0, -121.0)

    # Load station metadata from IEM data
    station_metadata = StationMetadata.from_iem_data(
        data_dir / "iem",
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
    )

    # Load station observations
    station_loader = IEMLoader(
        data_dir / "iem",
        year=year,
        station_metadata=station_metadata,
    )
    station_loader.load_to_memory()

    # Load grid data
    grid_loader = None
    if use_hrrr:
        grid_loader = HRRRLoader(
            data_dir / "hrrr",
            years=[year],
            lat_bounds=lat_bounds,
            lon_bounds=lon_bounds,
        )
    elif use_era5:
        grid_loader = ERA5Loader(
            data_dir / "era5",
            years=[year],
            lat_bounds=lat_bounds,
            lon_bounds=lon_bounds,
        )

    if grid_loader is not None:
        grid_loader.load_to_memory()

    # Create dataset
    dataset = WeatherDataset(
        year=year,
        back_hrs=back_hrs,
        lead_hours=lead_hours,
        station_metadata=station_metadata,
        station_loader=station_loader,
        grid_loader=grid_loader,
    )

    # Split into train/val
    n_samples = len(dataset)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    # Use contiguous split (later data for validation)
    train_dataset = torch.utils.data.Subset(dataset, range(n_train))
    val_dataset = torch.utils.data.Subset(dataset, range(n_train, n_samples))

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
