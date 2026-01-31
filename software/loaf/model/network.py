"""Graph network construction for weather station relationships.

Ported from LocalizedWeather: Network/MadisNetwork.py, ERA5Network.py, HRRRNetwork.py
Authors: Qidong Yang & Jonathan Giezendanner (original)

This module constructs graphs connecting:
- Stations to stations (internal graph)
- Grid points to stations (external graphs for ERA5/HRRR)
"""

import itertools
from enum import Enum

import numpy as np
import torch
from scipy.spatial import Delaunay
from torch_geometric.nn import knn, knn_graph


class NetworkConstructionMethod(Enum):
    """Method for constructing the station-to-station graph."""

    NONE = 0
    KNN = 1
    DELAUNAY = 2
    FULLY_CONNECTED = 3


def search_k_neighbors(
    target_points: torch.Tensor,
    source_points: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Find k nearest neighbors from source to target points.

    Args:
        target_points: Points to find neighbors for (n_target, n_features)
        source_points: Points to search in (n_source, n_features)
        k: Number of neighbors

    Returns:
        Edge index (2, n_edges) where edges go from source to target
    """
    edge_index = knn(source_points, target_points, k)[[1, 0], :]
    return edge_index


class StationNetwork:
    """Network connecting weather stations.

    Builds a graph connecting MADIS/IEM stations based on their
    geographic locations using KNN, Delaunay triangulation, or
    full connectivity.

    Args:
        station_lons: Station longitudes (n_stations,)
        station_lats: Station latitudes (n_stations,)
        n_neighbors: Number of neighbors for KNN
        method: Graph construction method
    """

    def __init__(
        self,
        station_lons: np.ndarray,
        station_lats: np.ndarray,
        n_neighbors: int = 5,
        method: NetworkConstructionMethod = NetworkConstructionMethod.KNN,
    ):
        self.n_neighbors = n_neighbors
        self.method = method

        # Convert to tensors
        self.stat_lons = torch.from_numpy(station_lons.astype(np.float32))
        self.stat_lats = torch.from_numpy(station_lats.astype(np.float32))
        self.n_stations = len(station_lons)

        # Station positions (n_stations, 2)
        self.pos = torch.stack([self.stat_lons, self.stat_lats], dim=1)

        # Longitude/latitude as separate tensors for model input
        self.madis_lon = self.stat_lons.reshape(-1, 1)
        self.madis_lat = self.stat_lats.reshape(-1, 1)

        # Build the graph
        self.edge_index = self._build_network()

    def _build_network(self) -> torch.Tensor:
        """Build station-to-station edge index."""
        if self.method == NetworkConstructionMethod.NONE:
            return torch.empty((2, 0), dtype=torch.long)

        elif self.method == NetworkConstructionMethod.KNN:
            return knn_graph(
                self.pos,
                k=self.n_neighbors,
                batch=torch.zeros(self.n_stations),
                loop=False,
            )

        elif self.method == NetworkConstructionMethod.DELAUNAY:
            # Delaunay triangulation creates triangle mesh
            tri = Delaunay(self.pos.numpy())
            simplices = tri.simplices

            # Extract edges from triangles
            edges = np.concatenate([
                simplices[:, [0, 1]],
                simplices[:, [1, 2]],
                simplices[:, [2, 0]],
            ])
            # Add reverse edges for undirected graph
            edges = np.concatenate([edges, np.flip(edges, axis=1)])
            # Remove duplicates
            edges = np.unique(edges, axis=0)
            edges = np.moveaxis(edges, 0, 1)

            return torch.from_numpy(edges).long()

        elif self.method == NetworkConstructionMethod.FULLY_CONNECTED:
            # All pairs of stations
            edges = list(itertools.permutations(range(self.n_stations), 2))
            return torch.tensor(edges, dtype=torch.long).t()

        else:
            raise ValueError(f"Unknown method: {self.method}")


class GridNetwork:
    """Network connecting grid points to weather stations.

    Builds a graph connecting grid points (ERA5 or HRRR) to nearby
    weather stations using KNN.

    Args:
        grid_lons: Grid longitudes (n_grid,) or (lat, lon)
        grid_lats: Grid latitudes (n_grid,) or (lat, lon)
        station_network: StationNetwork to connect to
        n_neighbors: Number of grid neighbors per station
    """

    def __init__(
        self,
        grid_lons: np.ndarray,
        grid_lats: np.ndarray,
        station_network: StationNetwork,
        n_neighbors: int = 4,
    ):
        self.n_neighbors = n_neighbors
        self.station_network = station_network

        # Flatten grid if 2D
        if grid_lons.ndim == 2:
            grid_lons = grid_lons.flatten()
            grid_lats = grid_lats.flatten()

        # Convert to tensors
        self.lons = torch.from_numpy(grid_lons.astype(np.float32))
        self.lats = torch.from_numpy(grid_lats.astype(np.float32))
        self.n_grid = len(self.lons)

        # Grid positions (n_grid, 2)
        self.pos = torch.stack([self.lons, self.lats], dim=1)

        # Build edge index from grid to stations
        self.edge_index = self._build_network()

    def _build_network(self) -> torch.Tensor:
        """Build grid-to-station edge index."""
        return search_k_neighbors(
            self.station_network.pos,
            self.pos,
            self.n_neighbors,
        ).long()


def build_networks(
    station_lons: np.ndarray,
    station_lats: np.ndarray,
    era5_lons: np.ndarray | None = None,
    era5_lats: np.ndarray | None = None,
    hrrr_lons: np.ndarray | None = None,
    hrrr_lats: np.ndarray | None = None,
    n_neighbors_m2m: int = 5,
    n_neighbors_e2m: int = 4,
    n_neighbors_h2m: int = 4,
    method: NetworkConstructionMethod = NetworkConstructionMethod.KNN,
) -> tuple[StationNetwork, GridNetwork | None, GridNetwork | None]:
    """Build all networks for the model.

    Args:
        station_lons: Station longitudes
        station_lats: Station latitudes
        era5_lons: ERA5 grid longitudes (optional)
        era5_lats: ERA5 grid latitudes (optional)
        hrrr_lons: HRRR grid longitudes (optional)
        hrrr_lats: HRRR grid latitudes (optional)
        n_neighbors_m2m: Station-to-station neighbors
        n_neighbors_e2m: ERA5-to-station neighbors
        n_neighbors_h2m: HRRR-to-station neighbors
        method: Graph construction method for stations

    Returns:
        Tuple of (station_network, era5_network, hrrr_network)
    """
    # Build station network
    station_network = StationNetwork(
        station_lons=station_lons,
        station_lats=station_lats,
        n_neighbors=n_neighbors_m2m,
        method=method,
    )

    # Build ERA5 network if provided
    era5_network = None
    if era5_lons is not None and era5_lats is not None:
        era5_network = GridNetwork(
            grid_lons=era5_lons,
            grid_lats=era5_lats,
            station_network=station_network,
            n_neighbors=n_neighbors_e2m,
        )

    # Build HRRR network if provided
    hrrr_network = None
    if hrrr_lons is not None and hrrr_lats is not None:
        hrrr_network = GridNetwork(
            grid_lons=hrrr_lons,
            grid_lats=hrrr_lats,
            station_network=station_network,
            n_neighbors=n_neighbors_h2m,
        )

    return station_network, era5_network, hrrr_network
