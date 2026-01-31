"""Graph Neural Network modules for weather forecasting.

This package contains the GNN components ported from LocalizedWeather:
- MPNN: Main message passing neural network
- GNNLayerInternal: Station-to-station message passing
- GNNLayerExternal: Grid-to-station message passing
"""

from loaf.model.gnn.external import GNNLayerExternal
from loaf.model.gnn.internal import GNNLayerInternal
from loaf.model.gnn.mpnn import MPNN

__all__ = ["MPNN", "GNNLayerInternal", "GNNLayerExternal"]
