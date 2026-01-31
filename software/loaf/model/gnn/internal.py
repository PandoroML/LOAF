"""Internal GNN layer for station-to-station message passing.

Ported from LocalizedWeather: Modules/GNN/GNN_Layer_Internal.py
Authors: Qidong Yang & Jonathan Giezendanner (original)

This layer implements message passing between weather stations (MADIS nodes),
following formula 8-9 of the LocalizedWeather paper.
"""

import torch
from torch import nn
from torch_geometric.nn import InstanceNorm, MessagePassing

from loaf.model.activations import Tanh


class GNNLayerInternal(MessagePassing):
    """Internal message passing layer for station-to-station communication.

    This layer propagates information between weather stations using a
    learned message function and node update function.

    Args:
        in_dim: Input feature dimension
        out_dim: Output feature dimension
        hidden_dim: Hidden layer dimension
        org_in_dim: Original input dimension (for message computation)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        org_in_dim: int,
    ):
        super().__init__(node_dim=-2, aggr="mean")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        # Message network: combines sender/receiver features + original features + positions
        # Input: x_i, x_j, u_i - u_j, pos_i - pos_j
        self.message_net_1 = nn.Sequential(
            nn.Linear(2 * in_dim + org_in_dim + 2, hidden_dim),
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
        x: torch.Tensor,
        u: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate messages along edges.

        Args:
            x: Node features (n_nodes, in_dim)
            u: Original node features (n_nodes, org_in_dim)
            pos: Node positions (n_nodes, 2) - [lon, lat]
            edge_index: Edge indices (2, n_edges)
            batch: Batch indices (n_nodes,)

        Returns:
            Updated node features (n_nodes, out_dim)
        """
        x = self.propagate(edge_index, x=x, u=u, pos=pos)
        x = self.norm(x, batch)
        return x

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        u_i: torch.Tensor,
        u_j: torch.Tensor,
        pos_i: torch.Tensor,
        pos_j: torch.Tensor,
    ) -> torch.Tensor:
        """Compute messages from neighbors.

        Following formula 8 of the paper:
        m_ij = MLP([x_i, x_j, u_i - u_j, pos_i - pos_j])

        Args:
            x_i: Receiver node features
            x_j: Sender node features
            u_i: Receiver original features
            u_j: Sender original features
            pos_i: Receiver positions
            pos_j: Sender positions

        Returns:
            Messages from j to i
        """
        message = self.message_net_1(
            torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j), dim=-1)
        )
        message = self.message_net_2(message)
        return message

    def update(self, message: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Update node features with aggregated messages.

        Following formula 9 of the paper:
        x_i = x_i + MLP([x_i, agg_j(m_ij)])

        Args:
            message: Aggregated messages
            x: Current node features

        Returns:
            Updated node features (with residual connection)
        """
        update = self.update_net_1(torch.cat((x, message), dim=-1))
        update = self.update_net_2(update)
        return x + update
