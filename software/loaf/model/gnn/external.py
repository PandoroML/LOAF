"""External GNN layer for grid-to-station message passing.

Ported from LocalizedWeather: Modules/GNN/GNN_Layer_External.py
Authors: Qidong Yang & Jonathan Giezendanner (original)

This layer implements message passing from grid points (ERA5/HRRR) to
weather stations (MADIS nodes), following formula 8-9 of the paper.
"""

import torch
from torch import nn
from torch_geometric.nn import MessagePassing, InstanceNorm

from loaf.model.activations import Tanh


class GNNLayerExternal(MessagePassing):
    """External message passing layer for grid-to-station communication.

    This layer propagates information from external grid points (ERA5/HRRR)
    to weather stations using a learned message function and node update.

    Args:
        in_dim: Input feature dimension for station nodes
        out_dim: Output feature dimension
        hidden_dim: Hidden layer dimension
        ex_in_dim: External input dimension (grid features)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        ex_in_dim: int,
    ):
        super().__init__(node_dim=-2, aggr="mean")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.ex_in_dim = ex_in_dim

        # External embedding network: embeds grid features + positions
        self.ex_embed_net_1 = nn.Sequential(
            nn.Linear(ex_in_dim + 2, hidden_dim),
            Tanh(),
        )
        self.ex_embed_net_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            Tanh(),
        )

        # Message network: combines station/grid features + position diff
        self.message_net_1 = nn.Sequential(
            nn.Linear(in_dim + hidden_dim + 2, hidden_dim),
            Tanh(),
        )
        self.message_net_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            Tanh(),
        )

        # Update network: combines node features with aggregated messages
        self.update_net_1 = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            Tanh(),
        )
        self.update_net_2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            Tanh(),
        )

        self.norm = InstanceNorm(out_dim)

    def forward(
        self,
        in_x: torch.Tensor,
        ex_x: torch.Tensor,
        in_pos: torch.Tensor,
        ex_pos: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate messages from grid to stations.

        Args:
            in_x: Station node features (n_stations, in_dim)
            ex_x: External grid features (n_grid, ex_in_dim)
            in_pos: Station positions (n_stations, 2)
            ex_pos: Grid positions (n_grid, 2)
            edge_index: Edge indices (2, n_edges) - from grid to stations
            batch: Batch indices (n_stations,)

        Returns:
            Updated station features (n_stations, out_dim)
        """
        n_in_x = in_x.size(0)

        # Embed external (grid) features with positions
        ex_x = self.ex_embed_net_1(torch.cat((ex_x, ex_pos), dim=1))
        ex_x = self.ex_embed_net_2(ex_x)

        # Concatenate station and grid nodes
        x = torch.cat((in_x, ex_x), dim=0)
        pos = torch.cat((in_pos, ex_pos), dim=0)

        # Shift edge indices to account for concatenation
        # Grid nodes are after station nodes in the concatenated tensor
        index_shift = torch.zeros_like(edge_index)
        index_shift[0] = index_shift[0] + n_in_x

        x = self.propagate(edge_index + index_shift, x=x, pos=pos)

        # Only return station node features
        x = x[:n_in_x]
        x = self.norm(x, batch)
        return x

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        pos_i: torch.Tensor,
        pos_j: torch.Tensor,
    ) -> torch.Tensor:
        """Compute messages from grid to stations.

        Following formula 8 of the paper:
        m_ij = MLP([x_i, x_j, pos_i - pos_j])

        Args:
            x_i: Receiver (station) features
            x_j: Sender (grid) features
            pos_i: Receiver positions
            pos_j: Sender positions

        Returns:
            Messages from grid to station
        """
        message = self.message_net_1(
            torch.cat((x_i, x_j, pos_i - pos_j), dim=-1)
        )
        message = self.message_net_2(message)
        return message

    def update(self, message: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Update station features with aggregated messages.

        Following formula 9 of the paper:
        x_i = x_i + MLP([x_i, agg_j(m_ij)])

        Args:
            message: Aggregated messages from grid
            x: Current station features

        Returns:
            Updated station features (with residual connection)
        """
        update = self.update_net_1(torch.cat((x, message), dim=-1))
        update = self.update_net_2(update)
        return x + update
