"""Per-layer normalization for the GNN layers, selectable by config.

The LocalizedWeather layers this package ports normalize each layer's output
with torch_geometric's InstanceNorm, which standardizes every feature channel
*across the nodes of each graph in the batch*. In the paper's setting (hundreds
of MADIS stations per graph) that is survivable. With a handful of stations it
is not: each channel's values across the 5 station nodes get re-centered to
mean 0 and rescaled to unit variance after every layer, which erases exactly
the information the model needs most - the absolute wind level that all
stations in the region share, including everything the grid-to-station layer
just pulled in from HRRR. Measured on 3 months of Arlington data (5 stations),
that left the model predicting the mean wind (vector RMSE 2.86 m/s at 6 h,
identical to climatology) with its output barely responding to the grid
input; switching the normalization off brought it to 1.71 m/s, ahead of the
raw HRRR forecast it is fed (1.91 m/s).

Options (model config key ``gnn_norm``):
  - "none": no per-layer normalization (default for new training runs).
  - "layer": nn.LayerNorm over each node's own feature vector; normalizes
    scale without mixing information across nodes. Scored close to "none".
  - "instance": the original cross-node InstanceNorm, kept for checkpoints
    trained before this option existed and for large-graph regions.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import InstanceNorm

GNN_NORM_TYPES = ("none", "layer", "instance")


class GNNNorm(nn.Module):
    """Wraps the chosen normalization behind one (x, batch) call signature."""

    def __init__(self, kind: str, dim: int):
        super().__init__()
        if kind not in GNN_NORM_TYPES:
            raise ValueError(f"Unknown gnn_norm: {kind!r} (expected one of {GNN_NORM_TYPES})")
        self.kind = kind
        self.norm: nn.Module | None
        if kind == "instance":
            self.norm = InstanceNorm(dim)
        elif kind == "layer":
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        if self.norm is None:
            return x
        if self.kind == "instance":
            return self.norm(x, batch)
        return self.norm(x)
