"""Model architecture modules for LOAF.

This package contains the neural network components for weather forecasting,
ported from the LocalizedWeather paper implementation.

Subpackages:
- gnn: Graph Neural Network components (MPNN, internal/external layers)
- transformer: Vision Transformer components (ViT, embeddings)

Modules:
- network: Graph construction utilities
- activations: Custom activation functions
"""

from loaf.model.activations import Swish, Tanh
from loaf.model.gnn import MPNN, GNNLayerExternal, GNNLayerInternal
from loaf.model.network import (
    GridNetwork,
    NetworkConstructionMethod,
    StationNetwork,
    build_networks,
)
from loaf.model.transformer import StationEmbed, Transformer, VisionTransformer

__all__ = [
    # GNN
    "MPNN",
    "GNNLayerInternal",
    "GNNLayerExternal",
    # Transformer
    "VisionTransformer",
    "StationEmbed",
    "Transformer",
    # Network
    "StationNetwork",
    "GridNetwork",
    "NetworkConstructionMethod",
    "build_networks",
    # Activations
    "Tanh",
    "Swish",
]
