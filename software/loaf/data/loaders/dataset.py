"""Combined weather dataset for training.

Combines ERA5/HRRR grid data with IEM/MADIS station observations
for the GNN + ViT model.

Adapted from LocalizedWeather MixData.py.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from dateutil import rrule
from torch.utils.data import DataLoader, Dataset, Subset

from .era5 import ERA5Loader
from .hrrr import HRRRLoader
from .iem import IEMLoader
from .stations import StationMetadata


class WeatherDataset(Dataset):
    """Combined dataset for weather forecasting.

    Combines gridded data (ERA5 or HRRR) with station observations
    for training the GNN + ViT model.

    Each sample splits a time window into a historical input (the
    ``back_hrs`` hours up to and including "now") and a multi-horizon
    target: ``target_vars`` at each offset in ``lead_times`` hours
    after "now".

    Args:
        year: Year of data to load.
        back_hrs: Number of historical hours for input.
        station_metadata: StationMetadata instance.
        station_loader: IEMLoader or MADISLoader instance.
        lead_times: Forecast horizons in hours (e.g. [6, 12, 24, 48]).
        grid_loader: ERA5Loader or HRRRLoader instance (optional).
        station_vars: List of station variables to use as input.
        grid_vars: List of grid variables to use as input.
        target_vars: List of variables to predict (default: u, v wind).
        normalize: Whether to apply min-max normalization.
    """

    def __init__(
        self,
        year: int,
        back_hrs: int,
        station_metadata: StationMetadata,
        station_loader: IEMLoader,
        lead_times: list[int] | None = None,
        grid_loader: ERA5Loader | HRRRLoader | None = None,
        station_vars: list[str] | None = None,
        grid_vars: list[str] | None = None,
        target_vars: list[str] | None = None,
        normalize: bool = True,
    ):
        self.year = year
        self.back_hrs = back_hrs
        self.lead_times = sorted(lead_times or [48])
        self.lead_hours = self.lead_times[-1]

        self.station_metadata = station_metadata
        self.station_loader = station_loader
        self.grid_loader = grid_loader

        # Default variables
        self.station_vars = station_vars or ["u", "v", "temp", "dewpoint"]
        self.grid_vars = grid_vars or ["u", "v", "temp", "dewpoint"]
        self.target_vars = target_vars or ["u", "v"]

        # Vars to fetch from the station loader: inputs + any targets not
        # already covered by station_vars.
        self._fetch_vars = list(
            dict.fromkeys([*self.station_vars, *self.target_vars])
        )

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
        """Generate hourly timeline for a year (timezone-naive for xarray compat)."""
        start = datetime(year, 1, 1)
        end = datetime(year + 1, 1, 1)
        times = list(rrule.rrule(rrule.HOURLY, dtstart=start, until=end))[:-1]
        return pd.DatetimeIndex(times)  # tz-naive for xarray compatibility

    def _compute_statistics(self) -> None:
        """Compute normalization statistics."""
        self.station_stats = self.station_loader.compute_statistics(self._fetch_vars)

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

    def _stack_vars(
        self,
        data: dict[str, torch.Tensor],
        variables: list[str],
        t_start: int,
        t_end: int,
        source: str,
    ) -> torch.Tensor:
        """Stack a list of variables into a (n_entities, n_time, n_vars) tensor."""
        values = []
        for var in variables:
            v = data[var][:, t_start:t_end]
            if self.normalize:
                v = self._normalize_var(v, var, source)
            values.append(v)
        return torch.stack(values, dim=-1)

    def _stack_masks(
        self,
        data: dict[str, torch.Tensor],
        variables: list[str],
        t_start: int,
        t_end: int,
    ) -> torch.Tensor:
        """Stack "is_real" masks for a list of variables into (n_entities, n_time, n_vars)."""
        masks = []
        for var in variables:
            key = f"{var}_is_real"
            if key in data:
                m = data[key][:, t_start:t_end]
            else:
                m = torch.ones_like(data[var][:, t_start:t_end])
            masks.append(m)
        return torch.stack(masks, dim=-1)

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        # We need back_hrs of history and lead_hours of future
        return len(self.timeline) - self.back_hrs - self.lead_hours

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a training sample.

        The time window covers ``back_hrs`` hours of history (ending at
        "now") plus ``max(lead_times)`` hours of future targets.

        Args:
            index: Sample index.

        Returns:
            Dictionary containing:
            - madis_x: Station input (n_stations, back_hrs, n_station_vars)
            - madis_lon, madis_lat: Normalized station coordinates (n_stations, 1)
            - edge_index: Station-to-station graph edges (2, n_edges)
            - target: (n_stations, n_lead_times, n_target_vars)
            - target_mask: "is_real" mask matching target, same shape
            - ex_x, ex_lon, ex_lat, edge_index_e2m: Grid inputs (if grid_loader given)
        """
        start_idx = index
        end_idx = index + self.back_hrs + self.lead_hours

        time_sel = self.timeline[start_idx : end_idx + 1]
        time_start = time_sel[0]
        time_end = time_sel[-1]

        sample: dict[str, Any] = {}

        # --- Station input: the back_hrs hours up to and including "now" ---
        station_data = self.station_loader.get_sample(time_start, time_end, self._fetch_vars)

        sample["madis_x"] = self._stack_vars(
            station_data, self.station_vars, 0, self.back_hrs, "station"
        )

        norm_lons, norm_lats = self._normalize_coords(
            self.station_metadata.lons, self.station_metadata.lats
        )
        sample["madis_lon"] = norm_lons.unsqueeze(-1)
        sample["madis_lat"] = norm_lats.unsqueeze(-1)
        sample["edge_index"] = self.station_metadata.get_k_edge_index()

        # --- Targets: target_vars at each lead time, offset from "now" ---
        target_steps = []
        target_mask_steps = []
        for lead_hr in self.lead_times:
            # "now" is relative index back_hrs - 1; lead_hr hours after that.
            t_idx = self.back_hrs - 1 + lead_hr
            target_steps.append(
                self._stack_vars(
                    station_data, self.target_vars, t_idx, t_idx + 1, "station"
                ).squeeze(1)
            )
            target_mask_steps.append(
                self._stack_masks(station_data, self.target_vars, t_idx, t_idx + 1).squeeze(1)
            )

        sample["target"] = torch.stack(target_steps, dim=1)  # (n_stations, n_lead_times, n_vars)
        sample["target_mask"] = torch.stack(target_mask_steps, dim=1)

        # --- Grid input (optional): historical window matching station input ---
        if self.grid_loader is not None:
            grid_data = self.grid_loader.get_sample(time_start, time_end, self.grid_vars)

            sample["ex_x"] = self._stack_vars(grid_data, self.grid_vars, 0, self.back_hrs, "grid")

            grid_pos = self.grid_loader.get_node_positions()
            norm_grid_lons, norm_grid_lats = self._normalize_coords(grid_pos[:, 0], grid_pos[:, 1])
            sample["ex_lon"] = norm_grid_lons.unsqueeze(-1)
            sample["ex_lat"] = norm_grid_lats.unsqueeze(-1)

            sample["edge_index_e2m"] = self._compute_grid_to_station_edges(
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

    def usable_index_range(self) -> tuple[int, int]:
        """Range [start, end) of sample indices safe to draw from.

        __len__ spans the whole calendar year regardless of how much of it
        was actually downloaded or observed. Two failure modes otherwise
        follow from that:

        - Station data is forward/backward-filled by IEMLoader, so it never
          crashes, but a chronological train/val split over the full
          __len__ range can put validation entirely past the last real
          observation (or before the first), i.e. all-imputed targets and
          meaningless masked metrics.
        - Grid (HRRR/ERA5) data is NOT filled - it's exactly whatever was
          downloaded. A sample whose input window falls outside the grid
          loader's covered time range gets an empty array back and crashes
          in get_sample()'s reshape.

        This bounds indices so every sample's targets land within real
        station observations, and (if a grid_loader is set) every sample's
        input window lands within the grid's downloaded time range.
        """
        ds = self.station_loader.data
        first_real_idx, last_real_idx = None, None
        for var in self.target_vars:
            key = f"{var}_is_real"
            if key not in ds.data_vars:
                continue
            real_at_time = ds[key].values.max(axis=0) > 0  # any station real, per time
            nonzero = real_at_time.nonzero()[0]
            if len(nonzero) == 0:
                continue
            lo, hi = int(nonzero[0]), int(nonzero[-1])
            first_real_idx = lo if first_real_idx is None else min(first_real_idx, lo)
            last_real_idx = hi if last_real_idx is None else max(last_real_idx, hi)

        if first_real_idx is None:
            return (0, 0)

        min_lead, max_lead = self.lead_times[0], self.lead_times[-1]
        start = max(0, first_real_idx - self.back_hrs + 1 - min_lead)
        end = last_real_idx - self.back_hrs - max_lead + 2  # exclusive

        if self.grid_loader is not None:
            grid_times = self.grid_loader.times
            if len(grid_times) == 0:
                return (0, 0)

            timeline_values = self.timeline.values
            grid_start_idx = int(np.searchsorted(timeline_values, grid_times.min(), side="left"))
            grid_end_idx = int(np.searchsorted(timeline_values, grid_times.max(), side="right")) - 1

            # Input window [index, index + back_hrs) must lie within grid coverage.
            # This is necessary but not sufficient if the grid has interior
            # gaps (e.g. a few missing hours within an otherwise-covered
            # range) - such samples can still raise a shape mismatch.
            start = max(start, grid_start_idx)
            end = min(end, grid_end_idx - self.back_hrs + 2)

        end = min(end, len(self))
        start = min(start, end)
        return (max(0, start), max(0, end))


@dataclass
class DatasetBundle:
    """Train/val dataloaders plus the underlying dataset for introspection."""

    train_loader: DataLoader
    val_loader: DataLoader
    dataset: WeatherDataset


def create_dataloaders(
    data_dir: str | Path,
    year: int,
    back_hrs: int = 24,
    lead_times: list[int] | None = None,
    batch_size: int = 32,
    val_split: float = 0.15,
    num_workers: int = 0,
    lat_bounds: tuple[float, float] | None = None,
    lon_bounds: tuple[float, float] | None = None,
    station_vars: list[str] | None = None,
    grid_vars: list[str] | None = None,
    target_vars: list[str] | None = None,
    min_observations: int = 24,
    use_era5: bool = False,
    use_hrrr: bool = False,
) -> DatasetBundle:
    """Create train and validation dataloaders.

    Args:
        data_dir: Base directory containing data subdirectories.
        year: Year of data to use.
        back_hrs: Number of historical hours.
        lead_times: Forecast horizons in hours (default: [48]).
        batch_size: Batch size for training.
        val_split: Fraction of data for validation.
        num_workers: Number of dataloader workers.
        lat_bounds: Geographic bounds (lat_min, lat_max).
        lon_bounds: Geographic bounds (lon_min, lon_max).
        station_vars: Station variables to use as input.
        grid_vars: Grid variables to use as input (if grid enabled).
        target_vars: Variables to predict (default: u, v wind).
        min_observations: Minimum real observations required to keep a station.
        use_era5: Whether to fuse ERA5 as grid data.
        use_hrrr: Whether to fuse HRRR as grid data.

    Returns:
        DatasetBundle with train_loader, val_loader, and the base dataset.
    """
    data_dir = Path(data_dir)

    # Default Seattle bounds
    if lat_bounds is None or lat_bounds[0] is None:
        lat_bounds = (46.5, 49.0)
    if lon_bounds is None or lon_bounds[0] is None:
        lon_bounds = (-124.0, -121.0)

    # Load station metadata from IEM data
    station_metadata = StationMetadata.from_iem_data(
        data_dir / "iem",
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        min_observations=min_observations,
    )

    if station_metadata.n_stations == 0:
        raise ValueError(
            f"No stations with >= {min_observations} observations found in "
            f"{data_dir / 'iem'} within bounds {lat_bounds}, {lon_bounds}."
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
        station_metadata=station_metadata,
        station_loader=station_loader,
        lead_times=lead_times,
        grid_loader=grid_loader,
        station_vars=station_vars,
        grid_vars=grid_vars,
        target_vars=target_vars,
    )

    # Restrict to samples with real (non-imputed) target coverage - and, if
    # grid fusion is on, within the grid loader's downloaded time range too -
    # so a chronological split doesn't push validation past the last real
    # observation, and grid sampling doesn't hit undownloaded time (see
    # usable_index_range() docstring).
    start_idx, end_idx = dataset.usable_index_range()
    n_samples = end_idx - start_idx
    if n_samples < 2:
        raise ValueError(
            f"Not enough real target/grid coverage for year {year} to build a "
            f"train/val split (back_hrs={back_hrs}, lead_hours={dataset.lead_hours}, "
            f"usable_samples={n_samples}). Try a different --year, download more "
            f"data, or reduce back_hrs/lead_times."
        )

    n_val = max(1, int(n_samples * val_split))
    n_train = n_samples - n_val

    # Use contiguous split (later data for validation)
    train_dataset = Subset(dataset, range(start_idx, start_idx + n_train))
    val_dataset = Subset(dataset, range(start_idx + n_train, end_idx))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return DatasetBundle(train_loader=train_loader, val_loader=val_loader, dataset=dataset)
