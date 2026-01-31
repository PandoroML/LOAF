"""Station metadata loader and graph construction.

Handles station location data and builds graphs for GNN message passing.
Adapted from LocalizedWeather MetaStation.py and Network modules.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import Delaunay


class StationMetadata:
    """Station metadata and graph construction.

    Loads station locations and builds graphs for the GNN layers.
    Supports both IEM (ASOS/AWOS) and MADIS station data.

    Args:
        data_path: Path to station data file (parquet or CSV with lat/lon).
        lat_bounds: Tuple of (lat_min, lat_max) for filtering.
        lon_bounds: Tuple of (lon_min, lon_max) for filtering.
        min_observations: Minimum observations required to include station.
        k_neighbors: Number of neighbors for KNN graph construction.
    """

    def __init__(
        self,
        data_path: str | Path | None = None,
        stations_df: pd.DataFrame | None = None,
        lat_bounds: tuple[float, float] | None = None,
        lon_bounds: tuple[float, float] | None = None,
        min_observations: int = 0,
        k_neighbors: int = 5,
    ):
        self.data_path = Path(data_path) if data_path else None
        self.k_neighbors = k_neighbors

        # Default to Seattle region
        if lat_bounds is None:
            lat_bounds = (46.5, 49.0)
        if lon_bounds is None:
            lon_bounds = (-124.0, -121.0)

        self.lat_min, self.lat_max = lat_bounds
        self.lon_min, self.lon_max = lon_bounds

        # Load or use provided station data
        if stations_df is not None:
            self._stations = stations_df
        elif data_path:
            self._stations = self._load_stations(min_observations)
        else:
            raise ValueError("Must provide either data_path or stations_df")

        # Filter to bounds
        self._stations = self._filter_to_bounds(self._stations)

        # Precompute graph
        self._k_edge_index: torch.Tensor | None = None
        self._delaunay_edge_index: torch.Tensor | None = None

    def _load_stations(self, min_observations: int) -> pd.DataFrame:
        """Load station data from file."""
        if self.data_path is None:
            raise ValueError("No data path provided")

        if self.data_path.suffix == ".parquet":
            df = pd.read_parquet(self.data_path)
        elif self.data_path.suffix == ".csv":
            df = pd.read_csv(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

        # Get unique stations with observation counts
        # Handle different column naming conventions
        lat_col = self._find_column(df, ["lat", "latitude"])
        lon_col = self._find_column(df, ["lon", "longitude"])
        station_col = self._find_column(df, ["station", "station_id", "id"])

        if station_col:
            station_counts = df.groupby(station_col).agg(
                {
                    lat_col: "first",
                    lon_col: "first",
                    station_col: "count",
                }
            )
            station_counts = station_counts.rename(columns={station_col: "n_obs"})
            station_counts = station_counts.reset_index()

            # Filter by observation count
            if min_observations > 0:
                station_counts = station_counts[station_counts["n_obs"] >= min_observations]

            return station_counts
        else:
            # No station column - just get unique lat/lon pairs
            unique_locs = df[[lat_col, lon_col]].drop_duplicates()
            unique_locs["station_id"] = range(len(unique_locs))
            return unique_locs

    def _find_column(
        self, df: pd.DataFrame, candidates: list[str]
    ) -> str | None:
        """Find a column name from a list of candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _filter_to_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter stations to geographic bounds."""
        lat_col = self._find_column(df, ["lat", "latitude"])
        lon_col = self._find_column(df, ["lon", "longitude"])

        mask = (
            (df[lat_col] >= self.lat_min)
            & (df[lat_col] <= self.lat_max)
            & (df[lon_col] >= self.lon_min)
            & (df[lon_col] <= self.lon_max)
        )
        return df[mask].reset_index(drop=True)

    @property
    def stations(self) -> pd.DataFrame:
        """Station metadata DataFrame."""
        return self._stations

    @property
    def n_stations(self) -> int:
        """Number of stations."""
        return len(self._stations)

    @property
    def positions(self) -> torch.Tensor:
        """Station positions as (lon, lat) tensor of shape (n_stations, 2)."""
        lat_col = self._find_column(self._stations, ["lat", "latitude"])
        lon_col = self._find_column(self._stations, ["lon", "longitude"])

        lons = self._stations[lon_col].values
        lats = self._stations[lat_col].values

        return torch.from_numpy(
            np.stack([lons, lats], axis=-1).astype(np.float32)
        )

    @property
    def lons(self) -> torch.Tensor:
        """Station longitudes."""
        lon_col = self._find_column(self._stations, ["lon", "longitude"])
        return torch.from_numpy(self._stations[lon_col].values.astype(np.float32))

    @property
    def lats(self) -> torch.Tensor:
        """Station latitudes."""
        lat_col = self._find_column(self._stations, ["lat", "latitude"])
        return torch.from_numpy(self._stations[lat_col].values.astype(np.float32))

    @property
    def station_ids(self) -> list[str]:
        """Station identifiers."""
        station_col = self._find_column(
            self._stations, ["station", "station_id", "id"]
        )
        if station_col:
            return self._stations[station_col].tolist()
        return [str(i) for i in range(len(self._stations))]

    def get_k_edge_index(self, k: int | None = None) -> torch.Tensor:
        """Build KNN graph edge index.

        Args:
            k: Number of neighbors. Defaults to self.k_neighbors.

        Returns:
            Edge index tensor of shape (2, n_edges) for use with PyG.
        """
        k = k or self.k_neighbors

        if self._k_edge_index is not None and k == self.k_neighbors:
            return self._k_edge_index

        positions = self.positions.numpy()
        n_stations = len(positions)

        # Compute pairwise distances
        # Using haversine would be more accurate, but Euclidean is fine for small regions
        diff = positions[:, None, :] - positions[None, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=-1))

        # For each station, find k nearest neighbors
        edge_src = []
        edge_dst = []

        for i in range(n_stations):
            # Sort by distance (excluding self)
            dist_i = distances[i].copy()
            dist_i[i] = np.inf  # Exclude self
            neighbors = np.argsort(dist_i)[:k]

            for j in neighbors:
                edge_src.append(i)
                edge_dst.append(j)

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

        if k == self.k_neighbors:
            self._k_edge_index = edge_index

        return edge_index

    def get_delaunay_edge_index(self) -> torch.Tensor:
        """Build Delaunay triangulation graph edge index.

        Returns:
            Edge index tensor of shape (2, n_edges) for use with PyG.
        """
        if self._delaunay_edge_index is not None:
            return self._delaunay_edge_index

        positions = self.positions.numpy()

        if len(positions) < 4:
            # Not enough points for Delaunay, use complete graph
            n = len(positions)
            edge_src = []
            edge_dst = []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        edge_src.append(i)
                        edge_dst.append(j)
            self._delaunay_edge_index = torch.tensor(
                [edge_src, edge_dst], dtype=torch.long
            )
            return self._delaunay_edge_index

        # Compute Delaunay triangulation
        tri = Delaunay(positions)

        # Extract edges from triangles
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((a, b))
                    edges.add((b, a))  # Bidirectional

        edge_src = [e[0] for e in edges]
        edge_dst = [e[1] for e in edges]

        self._delaunay_edge_index = torch.tensor(
            [edge_src, edge_dst], dtype=torch.long
        )
        return self._delaunay_edge_index

    @classmethod
    def from_iem_data(
        cls,
        data_dir: str | Path,
        lat_bounds: tuple[float, float] | None = None,
        lon_bounds: tuple[float, float] | None = None,
        min_observations: int = 100,
        k_neighbors: int = 5,
    ) -> "StationMetadata":
        """Create StationMetadata from IEM observation files.

        Args:
            data_dir: Directory containing IEM parquet files.
            lat_bounds: Geographic bounds for filtering.
            lon_bounds: Geographic bounds for filtering.
            min_observations: Minimum observations to include station.
            k_neighbors: Number of neighbors for KNN graph.

        Returns:
            StationMetadata instance.
        """
        data_dir = Path(data_dir)
        files = list(data_dir.glob("iem_*.parquet"))

        if not files:
            files = list(data_dir.glob("iem_*.csv"))

        if not files:
            raise FileNotFoundError(f"No IEM files found in {data_dir}")

        # Load all files and combine
        dfs = []
        for f in files:
            if f.suffix == ".parquet":
                dfs.append(pd.read_parquet(f))
            else:
                dfs.append(pd.read_csv(f))

        combined = pd.concat(dfs, ignore_index=True)

        # Get unique stations with counts
        station_col = None
        for col in ["station", "station_id", "id"]:
            if col in combined.columns:
                station_col = col
                break

        if station_col is None:
            raise ValueError("Cannot find station column in IEM data")

        lat_col = "lat" if "lat" in combined.columns else "latitude"
        lon_col = "lon" if "lon" in combined.columns else "longitude"

        # Count observations per station
        station_counts = combined.groupby(station_col).agg(
            n_obs=pd.NamedAgg(column=station_col, aggfunc="count"),
            lat=pd.NamedAgg(column=lat_col, aggfunc="first"),
            lon=pd.NamedAgg(column=lon_col, aggfunc="first"),
        ).reset_index()

        # Filter by count
        station_counts = station_counts[station_counts["n_obs"] >= min_observations]

        return cls(
            stations_df=station_counts.rename(columns={station_col: "station_id"}),
            lat_bounds=lat_bounds,
            lon_bounds=lon_bounds,
            k_neighbors=k_neighbors,
        )


def search_k_neighbors(
    query_pos: torch.Tensor,
    key_pos: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Find k-nearest neighbors from key positions for each query position.

    This is a utility function matching LocalizedWeather's NetworkUtils.

    Args:
        query_pos: Query positions of shape (n_query, 2).
        key_pos: Key positions of shape (n_key, 2).
        k: Number of neighbors to find.

    Returns:
        Indices of shape (1, n_query * k) containing the k nearest
        key indices for each query point.
    """
    # Compute pairwise distances
    diff = query_pos.unsqueeze(1) - key_pos.unsqueeze(0)  # (n_query, n_key, 2)
    distances = torch.sqrt(torch.sum(diff ** 2, dim=-1))  # (n_query, n_key)

    # Find k nearest for each query
    _, indices = torch.topk(distances, k, dim=1, largest=False)

    return indices.flatten().unsqueeze(0)
