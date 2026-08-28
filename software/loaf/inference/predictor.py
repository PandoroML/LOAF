"""Real-time inference for LOAF weather forecasting models.

Loads a checkpoint written by loaf.training.Trainer plus whatever station
(and optionally grid) data has been downloaded to disk, and produces
multi-horizon wind forecasts. Since the MPNN/ViT models predict jointly for
every station in their graph, Predictor always runs one forward pass over
the full station set and lets callers query the forecast for the station
nearest an arbitrary (lat, lon) - see predict().

Checkpoints from loaf.training.Trainer are self-describing: architecture
config, region bounds, and station normalization stats are embedded via
"run_config" / "station_stats", so Predictor only needs the checkpoint path
and a data directory - not the original training YAML.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from loaf.data.loaders import ERA5Loader, HRRRLoader, IEMLoader, StationMetadata
from loaf.training.trainer import build_model

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points, in kilometers."""
    lat1_r, lon1_r, lat2_r, lon2_r = (math.radians(v) for v in (lat1, lon1, lat2, lon2))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


@dataclass
class StationForecast:
    """Multi-horizon forecast for a single station.

    Attributes:
        station_id: Station identifier (e.g. IEM/ASOS code).
        lat, lon: Station coordinates.
        distance_km: Distance from the coordinate that was queried, if any
            (0.0 when this forecast wasn't produced by a nearest-station lookup).
        valid_time: The "now" timestamp the input window ends at - lead_hr
            hours after this is when each forecast value is valid for.
        lead_times: Forecast horizons in hours.
        target_vars: Names of the predicted variables.
        values: lead_hr -> {var: value}, in each variable's physical units.
    """

    station_id: str
    lat: float
    lon: float
    distance_km: float
    valid_time: pd.Timestamp
    lead_times: list[int]
    target_vars: list[str]
    values: dict[int, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation, used directly by the REST API."""
        return {
            "station_id": self.station_id,
            "lat": self.lat,
            "lon": self.lon,
            "distance_km": round(self.distance_km, 2),
            "valid_time": self.valid_time.isoformat(),
            "forecasts": [
                {"lead_hr": lead_hr, **self.values[lead_hr]} for lead_hr in self.lead_times
            ],
        }


class Predictor:
    """Loads a trained checkpoint and produces live multi-horizon forecasts.

    Args:
        checkpoint_path: Path to a checkpoint written by Trainer (best.pt/last.pt).
        data_dir: Base directory containing downloaded data (iem/, hrrr/, era5/).
        year: Year of station data to load. Defaults to the latest year with
            downloaded IEM data in data_dir.
        cache_ttl: Seconds to reuse a forward pass before recomputing it on
            the next predict()/predict_all() call.
        device: Torch device string. Defaults to CUDA if available, else CPU.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        data_dir: str | Path = "data",
        year: int | None = None,
        cache_ttl: float = 300.0,
        device: str | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.data_dir = Path(data_dir)
        self.cache_ttl = cache_ttl
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model_type: str = checkpoint["model_type"]
        self.lead_times: list[int] = checkpoint["lead_times"]
        self.target_vars: list[str] = checkpoint["target_vars"]
        self.station_vars: list[str] = checkpoint["station_vars"]
        # station_stats (full input coverage) is preferred; fall back to the
        # target-only stats saved by older checkpoints.
        self.station_stats: dict[str, dict[str, float]] = (
            checkpoint.get("station_stats") or checkpoint.get("target_stats") or {}
        )
        self.n_lead_times = len(self.lead_times)
        self.n_target_vars = len(self.target_vars)

        run_config: dict[str, Any] = checkpoint.get("run_config") or {}
        self.model_cfg: dict[str, Any] = dict(run_config.get("model_cfg") or {})
        self.model_cfg.setdefault("type", self.model_type)
        self.back_hrs: int = run_config.get("back_hrs", 24)
        self.lat_bounds: tuple[float, float] = tuple(
            run_config.get("lat_bounds") or (46.5, 49.0)
        )
        self.lon_bounds: tuple[float, float] = tuple(
            run_config.get("lon_bounds") or (-124.0, -121.0)
        )
        self.min_observations: int = run_config.get("min_observations", 24)
        self.use_hrrr: bool = bool(run_config.get("use_hrrr", False))
        self.use_era5: bool = bool(run_config.get("use_era5", False))
        self.grid_vars: list[str] | None = run_config.get("grid_vars")

        # Vars to fetch from the station loader: inputs + any targets not
        # already covered by station_vars (mirrors WeatherDataset).
        self._fetch_vars = list(dict.fromkeys([*self.station_vars, *self.target_vars]))

        self.year = year or self._infer_year()
        self.grid_loader: HRRRLoader | ERA5Loader | None = None
        self._load_data()
        self._n_stations = self.station_metadata.n_stations

        self.model = build_model(
            self.model_cfg,
            n_stations=self._n_stations,
            in_hrs=self.back_hrs,
            n_station_vars=len(self.station_vars),
            n_out_features=self.n_lead_times * self.n_target_vars,
            grid_vars=self.grid_vars if self.grid_loader is not None else None,
            in_hrs_grid=self.back_hrs,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self._cache: list[StationForecast] | None = None
        self._cache_at: float = 0.0

    @property
    def n_stations(self) -> int:
        """Number of stations the loaded model was built for."""
        return self._n_stations

    def _infer_year(self) -> int:
        """Latest year with downloaded IEM data in data_dir."""
        iem_dir = self.data_dir / "iem"
        files = sorted(iem_dir.glob("iem_*.parquet")) + sorted(iem_dir.glob("iem_*.csv"))
        if not files:
            raise FileNotFoundError(
                f"No IEM data found in {iem_dir} - download data first (see "
                f"loaf-download-iem) or pass year explicitly."
            )
        return max(int(f.stem.split("_")[1]) for f in files)

    def _load_data(self) -> None:
        """(Re)load station (and grid, if enabled) data from disk."""
        station_metadata = StationMetadata.from_iem_data(
            self.data_dir / "iem",
            lat_bounds=self.lat_bounds,
            lon_bounds=self.lon_bounds,
            min_observations=self.min_observations,
        )
        if station_metadata.n_stations == 0:
            raise ValueError(
                f"No stations with >= {self.min_observations} observations found in "
                f"{self.data_dir / 'iem'} within bounds {self.lat_bounds}, {self.lon_bounds}."
            )

        station_loader = IEMLoader(
            self.data_dir / "iem",
            year=self.year,
            station_metadata=station_metadata,
            variables=self._fetch_vars,
        )
        station_loader.load_to_memory()

        grid_loader: HRRRLoader | ERA5Loader | None = None
        if self.use_hrrr:
            grid_loader = HRRRLoader(
                self.data_dir / "hrrr",
                years=[self.year],
                lat_bounds=self.lat_bounds,
                lon_bounds=self.lon_bounds,
                variables=self.grid_vars,
            )
            grid_loader.load_to_memory()
        elif self.use_era5:
            grid_loader = ERA5Loader(
                self.data_dir / "era5",
                years=[self.year],
                lat_bounds=self.lat_bounds,
                lon_bounds=self.lon_bounds,
                variables=self.grid_vars,
            )
            grid_loader.load_to_memory()

        self.station_metadata = station_metadata
        self.station_loader = station_loader
        self.grid_loader = grid_loader

        norm_lons, norm_lats = self._normalize_coords(station_metadata.lons, station_metadata.lats)
        self._norm_lons = norm_lons.unsqueeze(-1)
        self._norm_lats = norm_lats.unsqueeze(-1)
        self._edge_index = station_metadata.get_k_edge_index()

    def refresh_data(self) -> None:
        """Reload station/grid data from disk, e.g. after a new hourly download.

        Refuses to swap in a station set of a different size than the one the
        model was built for - the graph and (for ViT) positional embeddings
        are sized to n_stations at construction time - and keeps serving the
        previous data instead. Retrain to pick up a changed station set.
        """
        previous = (
            self.station_metadata,
            self.station_loader,
            self.grid_loader,
            self._norm_lons,
            self._norm_lats,
            self._edge_index,
            self.year,
        )
        try:
            self.year = self._infer_year()
            self._load_data()
        except Exception:
            (
                self.station_metadata,
                self.station_loader,
                self.grid_loader,
                self._norm_lons,
                self._norm_lats,
                self._edge_index,
                self.year,
            ) = previous
            raise

        if self.station_metadata.n_stations != self._n_stations:
            new_count = self.station_metadata.n_stations
            (
                self.station_metadata,
                self.station_loader,
                self.grid_loader,
                self._norm_lons,
                self._norm_lats,
                self._edge_index,
                self.year,
            ) = previous
            raise RuntimeError(
                f"Station count changed ({self._n_stations} -> {new_count}); keeping "
                "previous data. Retrain to pick up the new station set."
            )

        self._cache = None

    def _normalize_coords(
        self, lons: torch.Tensor, lats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize coordinates to [0, 1] using the trained region's bounds."""
        eps = 1e-5
        norm_lons = (lons - self.lon_bounds[0]) / (self.lon_bounds[1] - self.lon_bounds[0] + eps)
        norm_lats = (lats - self.lat_bounds[0]) / (self.lat_bounds[1] - self.lat_bounds[0] + eps)
        return norm_lons, norm_lats

    def _normalize_var(self, values: torch.Tensor, var: str) -> torch.Tensor:
        stats = self.station_stats.get(var)
        if stats is None:
            return values
        eps = 1e-5
        return (values - stats["min"]) / (stats["max"] - stats["min"] + eps)

    def _denormalize_var(self, value: float, var: str) -> float:
        stats = self.station_stats.get(var)
        if stats is None:
            return value
        eps = 1e-5
        return value * (stats["max"] - stats["min"] + eps) + stats["min"]

    def _stack_vars(self, data: dict[str, torch.Tensor], variables: list[str]) -> torch.Tensor:
        """Stack variables into a (n_entities, n_time, n_vars) tensor, normalized."""
        return torch.stack([self._normalize_var(data[var], var) for var in variables], dim=-1)

    def _latest_window(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """(start, end) of the most recent back_hrs-hour window with real station data."""
        ds = self.station_loader.data
        real_any = None
        for var in self._fetch_vars:
            key = f"{var}_is_real"
            if key not in ds.data_vars:
                continue
            real = ds[key].values.max(axis=0) > 0
            real_any = real if real_any is None else (real_any | real)

        if real_any is None or not real_any.any():
            raise RuntimeError(
                f"No real station observations in {self.data_dir / 'iem'} for {self.year} - "
                "cannot build an inference window."
            )

        last_idx = int(np.nonzero(real_any)[0][-1])
        if last_idx + 1 < self.back_hrs:
            raise RuntimeError(
                f"Only {last_idx + 1} hour(s) of station data available, need "
                f"back_hrs={self.back_hrs}."
            )

        times = self.station_loader.timeline
        return times[last_idx - self.back_hrs + 1], times[last_idx]

    def _grid_to_station_edges(self, grid_pos: torch.Tensor, k: int = 4) -> torch.Tensor:
        station_pos = self.station_metadata.positions
        diff = station_pos.unsqueeze(1) - grid_pos.unsqueeze(0)
        distances = torch.sqrt(torch.sum(diff**2, dim=-1))
        _, indices = torch.topk(distances, min(k, grid_pos.shape[0]), dim=1, largest=False)
        edge_src = indices.flatten()
        edge_dst = torch.arange(station_pos.shape[0]).repeat_interleave(indices.shape[1])
        return torch.stack([edge_src, edge_dst], dim=0)

    @torch.no_grad()
    def predict_all(self, force_refresh: bool = False) -> list[StationForecast]:
        """Run one forward pass, returning a forecast for every station.

        Results are cached for cache_ttl seconds so repeated API requests
        don't each re-run the model; pass force_refresh=True to bypass.
        """
        now = time.monotonic()
        cache_fresh = self._cache is not None and (now - self._cache_at) < self.cache_ttl
        if not force_refresh and cache_fresh:
            return self._cache

        time_start, time_end = self._latest_window()
        station_data = self.station_loader.get_sample(time_start, time_end, self._fetch_vars)

        madis_x = self._stack_vars(station_data, self.station_vars).unsqueeze(0).to(self.device)
        madis_lon = self._norm_lons.unsqueeze(0).to(self.device)
        madis_lat = self._norm_lats.unsqueeze(0).to(self.device)
        edge_index = self._edge_index.unsqueeze(0).to(self.device)

        ex_x = ex_lon = ex_lat = edge_index_e2m = None
        if self.grid_loader is not None:
            grid_data = self.grid_loader.get_sample(time_start, time_end, self.grid_vars)
            ex_x = self._stack_vars(grid_data, self.grid_vars).unsqueeze(0).to(self.device)
            grid_pos = self.grid_loader.get_node_positions()
            norm_grid_lon, norm_grid_lat = self._normalize_coords(grid_pos[:, 0], grid_pos[:, 1])
            ex_lon = norm_grid_lon.unsqueeze(0).unsqueeze(-1).to(self.device)
            ex_lat = norm_grid_lat.unsqueeze(0).unsqueeze(-1).to(self.device)
            edge_index_e2m = self._grid_to_station_edges(grid_pos).unsqueeze(0).to(self.device)

        if self.model_type == "mpnn":
            preds = self.model(
                madis_x, madis_lon, madis_lat, edge_index, ex_lon, ex_lat, ex_x, edge_index_e2m
            )
        else:  # vit
            preds, _ = self.model(madis_x, era5_x=ex_x)

        n_batch, n_stations, _ = preds.shape
        preds = preds.view(n_batch, n_stations, self.n_lead_times, self.n_target_vars)[0].cpu()

        forecasts = []
        for i, station_id in enumerate(self.station_metadata.station_ids):
            values = {
                lead_hr: {
                    var: self._denormalize_var(float(preds[i, li, vi]), var)
                    for vi, var in enumerate(self.target_vars)
                }
                for li, lead_hr in enumerate(self.lead_times)
            }
            forecasts.append(
                StationForecast(
                    station_id=str(station_id),
                    lat=float(self.station_metadata.lats[i]),
                    lon=float(self.station_metadata.lons[i]),
                    distance_km=0.0,
                    valid_time=pd.Timestamp(time_end),
                    lead_times=self.lead_times,
                    target_vars=self.target_vars,
                    values=values,
                )
            )

        self._cache = forecasts
        self._cache_at = now
        return forecasts

    def predict(self, lat: float, lon: float, force_refresh: bool = False) -> StationForecast:
        """Forecast for the station nearest (lat, lon).

        Raises:
            ValueError: If (lat, lon) is more than 2x the trained region's
                diagonal away from the nearest station - i.e. well outside
                the model's coverage area.
        """
        forecasts = self.predict_all(force_refresh=force_refresh)
        for forecast in forecasts:
            forecast.distance_km = haversine_km(lat, lon, forecast.lat, forecast.lon)
        nearest = min(forecasts, key=lambda f: f.distance_km)

        region_diagonal_km = haversine_km(
            self.lat_bounds[0], self.lon_bounds[0], self.lat_bounds[1], self.lon_bounds[1]
        )
        if nearest.distance_km > 2 * region_diagonal_km:
            raise ValueError(
                f"({lat}, {lon}) is {nearest.distance_km:.0f} km from the nearest trained "
                f"station ({nearest.station_id}) - well outside the model's region."
            )
        return nearest
