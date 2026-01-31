"""Vision Transformer (ViT) for weather station predictions.

Ported from LocalizedWeather: Modules/Transformer/ViT.py
Authors: Qidong Yang & Jonathan Giezendanner (original)

This module applies a transformer architecture to weather station data,
treating each station as a token with its time series embedded.
"""


import torch
from torch import nn

from loaf.model.transformer.core import Transformer
from loaf.model.transformer.embeddings import StationEmbed


class VisionTransformer(nn.Module):
    """Vision Transformer for weather station predictions.

    Each station is treated as a "patch" (token) with its time series
    embedded. The transformer learns relationships between stations.

    Args:
        n_stations: Number of weather stations
        madis_len: Length of MADIS time series (hours)
        madis_n_vars_i: Number of MADIS input variables
        madis_n_vars_o: Number of output variables (predictions)
        dim: Embedding dimension
        attn_dim: Attention hidden dimension
        mlp_dim: FFN hidden dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        era5_n_vars: Number of ERA5 variables (optional)
        era5_len: Length of ERA5 time series (optional)
    """

    def __init__(
        self,
        n_stations: int,
        madis_len: int,
        madis_n_vars_i: int,
        madis_n_vars_o: int,
        dim: int,
        attn_dim: int,
        mlp_dim: int,
        num_heads: int,
        num_layers: int,
        era5_n_vars: int | None = None,
        era5_len: int | None = None,
    ):
        super().__init__()

        # Station embedding (time series → token)
        self.station_embed = StationEmbed(
            madis_len=madis_len,
            madis_n_vars_i=madis_n_vars_i,
            era5_len=era5_len,
            era5_n_vars=era5_n_vars,
            hidden_dim=dim,
        )

        # Learnable positional embedding for each station
        self.pos_E = nn.Embedding(n_stations, dim)

        # Transformer encoder
        self.transformer = Transformer(
            dim=dim,
            attn_dim=attn_dim,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

        # Output projection head
        self.head = nn.Sequential(
            nn.Linear(dim, madis_n_vars_o),
        )

    def forward(
        self,
        madis_x: torch.Tensor,
        era5_x: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass through the Vision Transformer.

        Args:
            madis_x: MADIS observations (n_batch, n_stations, madis_len, n_vars)
            era5_x: ERA5 data (n_batch, n_stations, era5_len, n_vars) or None
            return_attn: Whether to return attention weights

        Returns:
            out: Predictions (n_batch, n_stations, madis_n_vars_o)
            alphas: Attention weights (n_batch, num_layers, num_heads, n_stations, n_stations)
                   or None if return_attn=False
        """
        # Generate station embeddings from time series
        embs = self.station_embed(madis_x, era5_x)
        # (n_batch, n_stations, dim)

        B, T, _ = embs.shape

        # Add positional embeddings
        pos_ids = torch.arange(T).expand(B, -1).to(embs.device)
        embs = embs + self.pos_E(pos_ids)
        # (n_batch, n_stations, dim)

        # Apply transformer
        x, alphas = self.transformer(embs, attn_mask=None, return_attn=return_attn)
        # (n_batch, n_stations, dim)

        # Project to output
        out = self.head(x)
        # (n_batch, n_stations, madis_n_vars_o)

        return out, alphas
