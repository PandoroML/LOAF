"""Station embedding module for the Vision Transformer.

Ported from LocalizedWeather: Modules/Transformer/StationsEmbedding.py
Authors: Qidong Yang & Jonathan Giezendanner (original)

This module embeds weather station time series data into a fixed-dimensional
representation for use in the transformer architecture.
"""


import torch
from torch import nn


class StationEmbed(nn.Module):
    """Embed station time series data into transformer input tokens.

    Each weather variable (u, v, temp, etc.) is embedded separately using
    a linear layer, then all embeddings are concatenated and merged.

    Args:
        madis_len: Length of MADIS time series (number of hours)
        madis_n_vars_i: Number of MADIS input variables
        era5_len: Length of ERA5 time series (optional)
        era5_n_vars: Number of ERA5 variables (optional)
        hidden_dim: Output embedding dimension
    """

    def __init__(
        self,
        madis_len: int,
        madis_n_vars_i: int,
        era5_len: int | None = None,
        era5_n_vars: int | None = None,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.madis_len = madis_len
        self.hidden_dim = hidden_dim
        self.madis_n_vars_i = madis_n_vars_i
        self.era5_n_vars = era5_n_vars
        self.era5_len = era5_len

        # Each MADIS variable gets its own embedding layer
        # Input: time series of length madis_len
        # Output: hidden_dim features
        self.encoding_madis_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(madis_len, hidden_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in range(madis_n_vars_i)
            ]
        )

        # Calculate merge input dimension
        self.in_merge_dim = hidden_dim * madis_n_vars_i

        # Optional ERA5 embedding layers
        if era5_len is not None and era5_n_vars is not None:
            self.in_merge_dim += hidden_dim * era5_n_vars

            self.encoding_era5_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(era5_len, hidden_dim),
                        nn.ReLU(inplace=True),
                    )
                    for _ in range(era5_n_vars)
                ]
            )
        else:
            self.encoding_era5_layers = None

        # Merge all variable embeddings into final hidden_dim
        self.merge_net = nn.Sequential(
            nn.Linear(self.in_merge_dim, hidden_dim),
        )

    def forward(
        self,
        madis_x: torch.Tensor,
        era5_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed station data into transformer tokens.

        Args:
            madis_x: MADIS observations (n_batch, n_stations, madis_len, n_vars)
            era5_x: ERA5 data (n_batch, n_stations, era5_len, n_vars) or None

        Returns:
            Station embeddings (n_batch, n_stations, hidden_dim)
        """
        # Embed each MADIS variable separately, then concatenate
        # madis_x[:, :, :, i] has shape (n_batch, n_stations, madis_len)
        all_emb = torch.cat(
            [
                self.encoding_madis_layers[i](madis_x[:, :, :, i])
                for i in range(self.madis_n_vars_i)
            ],
            dim=-1,
        )
        # (n_batch, n_stations, hidden_dim * madis_n_vars_i)

        # Add ERA5 embeddings if provided
        if self.era5_len is not None and era5_x is not None:
            era5_emb = torch.cat(
                [
                    self.encoding_era5_layers[i](era5_x[:, :, :, i])
                    for i in range(self.era5_n_vars)
                ],
                dim=-1,
            )
            # (n_batch, n_stations, hidden_dim * era5_n_vars)
            all_emb = torch.cat((all_emb, era5_emb), dim=-1)

        # Merge all embeddings
        out = self.merge_net(all_emb)
        # (n_batch, n_stations, hidden_dim)

        return out
