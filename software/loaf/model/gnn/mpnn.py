"""Message Passing Neural Network (MPNN) for weather forecasting.

Ported from LocalizedWeather: Modules/GNN/MPNN.py
Authors: Qidong Yang & Jonathan Giezendanner (original)

This module implements the heterogeneous MPNN that combines internal
(station-to-station) and external (grid-to-station) message passing.
"""

from typing import Optional

import torch
from torch import nn
from torch_geometric.data import Data

from loaf.model.activations import Tanh
from loaf.model.gnn.external import GNNLayerExternal
from loaf.model.gnn.internal import GNNLayerInternal


class MPNN(nn.Module):
    """Message Passing Neural Network for weather station predictions.

    This network combines:
    1. Internal message passing (station-to-station)
    2. External message passing (grid-to-station from ERA5/HRRR)

    The architecture follows the LocalizedWeather paper:
    - External layer 1 (grid → stations)
    - N internal layers (station ↔ station)
    - External layer 2 (grid → stations)
    - Output MLP

    Args:
        n_passing: Number of internal message passing layers
        lead_hrs: Number of forecast lead hours (unused, kept for compatibility)
        n_node_features_m: Number of MADIS station features (flattened)
        n_node_features_e: Number of external grid features (flattened)
        n_out_features: Number of output features (e.g., 2 for u, v wind)
        hidden_dim: Hidden dimension for all layers
    """

    def __init__(
        self,
        n_passing: int,
        lead_hrs: int,
        n_node_features_m: int,
        n_node_features_e: int,
        n_out_features: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.lead_hrs = lead_hrs
        self.n_node_features_m = n_node_features_m
        self.n_node_features_e = n_node_features_e
        self.n_passing = n_passing
        self.hidden_dim = hidden_dim
        self.n_out_features = n_out_features

        # External layers (grid → station)
        self.gnn_ex_1 = GNNLayerExternal(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            ex_in_dim=n_node_features_e,
        )
        self.gnn_ex_2 = GNNLayerExternal(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            ex_in_dim=n_node_features_e,
        )

        # Internal layers (station ↔ station)
        self.gnn_layers = nn.ModuleList(
            [
                GNNLayerInternal(
                    in_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    out_dim=hidden_dim,
                    org_in_dim=n_node_features_m,
                )
                for _ in range(n_passing)
            ]
        )

        # Embedding MLP: project input features + position to hidden_dim
        self.embedding_mlp = nn.Sequential(
            nn.Linear(n_node_features_m + 2, hidden_dim),
            Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            Tanh(),
        )

        # Output MLP: project hidden features to predictions
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            Tanh(),
            nn.Linear(hidden_dim, n_out_features),
        )

    def build_graph_internal(
        self,
        x: torch.Tensor,
        madis_lon: torch.Tensor,
        madis_lat: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Data:
        """Build internal graph for station-to-station message passing.

        Args:
            x: Station features (n_batch, n_stations, n_features)
            madis_lon: Station longitudes (n_batch, n_stations, 1)
            madis_lat: Station latitudes (n_batch, n_stations, 1)
            edge_index: Edge indices (n_batch, 2, n_edges)

        Returns:
            PyG Data object with batched graph
        """
        n_batch = x.size(0)
        n_stations = x.size(1)

        # Flatten batch dimension
        x = x.view(n_batch * n_stations, -1)

        # Combine positions
        pos = torch.cat((madis_lon, madis_lat), dim=2)
        pos = pos.view(n_batch * n_stations, -1)

        # Create batch indices
        batch = torch.arange(n_batch).view(-1, 1) * torch.ones(1, n_stations)
        batch = batch.view(n_batch * n_stations).to(x.device)

        # Shift edge indices for batched graph
        index_shift = (torch.arange(n_batch) * n_stations).view(-1, 1, 1).to(x.device)
        edge_index = torch.cat(list(edge_index + index_shift), dim=1)

        return Data(x=x, pos=pos, batch=batch.long(), edge_index=edge_index.long())

    def build_graph_external(
        self,
        madis_x: torch.Tensor,
        ex_x: torch.Tensor,
        ex_lon: torch.Tensor,
        ex_lat: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Data:
        """Build external graph for grid-to-station message passing.

        Args:
            madis_x: Station features (n_batch, n_stations_m, n_features_m)
            ex_x: External grid features (n_batch, n_stations_e, n_features_e)
            ex_lon: Grid longitudes (n_batch, n_stations_e, 1)
            ex_lat: Grid latitudes (n_batch, n_stations_e, 1)
            edge_index: Edge indices (n_batch, 2, n_edges)

        Returns:
            PyG Data object with batched graph
        """
        n_batch = madis_x.size(0)
        n_stations_m = madis_x.size(1)
        n_stations_e = ex_x.size(1)

        # Flatten external features
        ex_x = ex_x.view(n_batch * n_stations_e, -1)

        # Combine external positions
        ex_pos = torch.cat(
            (
                ex_lon.view(n_batch, n_stations_e, 1),
                ex_lat.view(n_batch, n_stations_e, 1),
            ),
            dim=2,
        )
        ex_pos = ex_pos.view(n_batch * n_stations_e, -1)

        # Shift edge indices for batched graph
        madis_shift = (torch.arange(n_batch) * n_stations_m).view((n_batch, 1))
        ex_shift = (torch.arange(n_batch) * n_stations_e).view((n_batch, 1))
        shift = torch.cat((ex_shift, madis_shift), dim=1).unsqueeze(-1).to(madis_x.device)
        edge_index = torch.cat(list(edge_index + shift), dim=1)

        return Data(x=ex_x, pos=ex_pos, edge_index=edge_index.long())

    def forward(
        self,
        madis_x: torch.Tensor,
        madis_lon: torch.Tensor,
        madis_lat: torch.Tensor,
        edge_index: torch.Tensor,
        ex_lon: Optional[torch.Tensor],
        ex_lat: Optional[torch.Tensor],
        ex_x: Optional[torch.Tensor],
        edge_index_e2m: Optional[torch.Tensor],
        *args,
    ) -> torch.Tensor:
        """Forward pass through the MPNN.

        Args:
            madis_x: Station observations (n_batch, n_stations_m, n_hours_m, n_features_m)
            madis_lon: Station longitudes (n_batch, n_stations_m, 1)
            madis_lat: Station latitudes (n_batch, n_stations_m, 1)
            edge_index: Station-to-station edges (n_batch, 2, n_edges)
            ex_lon: Grid longitudes (n_batch, n_stations_e, 1) or None
            ex_lat: Grid latitudes (n_batch, n_stations_e, 1) or None
            ex_x: Grid features (n_batch, n_stations_e, n_hours_e, n_features_e) or None
            edge_index_e2m: Grid-to-station edges (n_batch, 2, n_edges) or None

        Returns:
            Predictions (n_batch, n_stations_m, n_out_features)
        """
        n_batch, n_stations_m, n_hours_m, n_features_m = madis_x.shape

        # Flatten time dimension
        madis_x = madis_x.view(n_batch, n_stations_m, -1)

        # Build internal graph
        in_graph = self.build_graph_internal(madis_x, madis_lon, madis_lat, edge_index)
        u = in_graph.x
        in_pos = in_graph.pos
        batch = in_graph.batch
        edge_index_m2m = in_graph.edge_index

        # Initial embedding
        in_x = self._forward_embedding_mlp(in_pos, u)

        # Prepare external graph if provided
        if ex_x is not None:
            b, n, t, v = ex_x.shape
            ex_x = ex_x.view(b, n, -1)
            ex_graph = self.build_graph_external(
                madis_x, ex_x, ex_lon, ex_lat, edge_index_e2m
            )
            ex_x = ex_graph.x
            ex_pos = ex_graph.pos
            edge_index_e2m = ex_graph.edge_index

        # External layer 1 (grid → stations)
        if ex_x is not None:
            in_x = self._forward_external_layer(
                self.gnn_ex_1, batch, edge_index_e2m, ex_pos, ex_x, in_pos, in_x
            )

        # Internal layers (station ↔ station)
        for i in range(self.n_passing):
            in_x = self._forward_internal_layer(
                batch, edge_index_m2m, i, in_pos, in_x, u
            )

        # External layer 2 (grid → stations)
        if ex_x is not None:
            in_x = self._forward_external_layer(
                self.gnn_ex_2, batch, edge_index_e2m, ex_pos, ex_x, in_pos, in_x
            )

        # Output projection
        out = self._forward_output_mlp(in_x)
        out = out.view(n_batch, n_stations_m, self.n_out_features)

        return out

    def _forward_output_mlp(self, in_x: torch.Tensor) -> torch.Tensor:
        """Apply output MLP."""
        return self.output_mlp(in_x)

    def _forward_embedding_mlp(
        self, in_pos: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """Apply embedding MLP to input features and positions."""
        return self.embedding_mlp(torch.cat((u, in_pos), dim=-1))

    def _forward_internal_layer(
        self,
        batch: torch.Tensor,
        edge_index_m2m: torch.Tensor,
        layer_idx: int,
        in_pos: torch.Tensor,
        in_x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Apply internal GNN layer."""
        return self.gnn_layers[layer_idx](in_x, u, in_pos, edge_index_m2m, batch)

    def _forward_external_layer(
        self,
        layer: GNNLayerExternal,
        batch: torch.Tensor,
        edge_index_e2m: torch.Tensor,
        ex_pos: torch.Tensor,
        ex_x: torch.Tensor,
        in_pos: torch.Tensor,
        in_x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply external GNN layer."""
        return layer(in_x, ex_x, in_pos, ex_pos, edge_index_e2m, batch)
