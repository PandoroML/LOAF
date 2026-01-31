"""Core Transformer components.

Ported from LocalizedWeather: Modules/Transformer/Transformer.py
Authors: Qidong Yang & Jonathan Giezendanner (original)

This module contains the attention head, multi-headed attention,
feed-forward network, and transformer encoder layers.
"""


import numpy as np
import torch
from torch import nn


class AttentionHead(nn.Module):
    """Single attention head.

    Args:
        dim: Input dimension
        n_hidden: Hidden dimension for keys, queries, and values
    """

    def __init__(self, dim: int, n_hidden: int):
        super().__init__()

        self.W_K = nn.Linear(dim, n_hidden)
        self.W_Q = nn.Linear(dim, n_hidden)
        self.W_V = nn.Linear(dim, n_hidden)
        self.n_hidden = n_hidden

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention.

        Args:
            x: Input tensor (B, T, dim)
            attn_mask: Attention mask (B, T, T) or None
                       1 = attend, 0 = mask out

        Returns:
            attn_output: Attention output (B, T, n_hidden)
            alpha: Attention weights (B, T, T)
        """
        q = self.W_Q(x)
        k = self.W_K(x)
        v = self.W_V(x)
        # (B, T, n_hidden)

        # Scaled dot-product attention
        pre_alpha = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(self.n_hidden)
        # (B, T, T)

        if attn_mask is not None:
            pre_alpha.masked_fill_(attn_mask == 0, -1e9)

        alpha = torch.softmax(pre_alpha, dim=-1)
        # (B, T, T)

        attn_output = torch.bmm(alpha, v)
        # (B, T, n_hidden)

        return attn_output, alpha


class MultiHeadedAttention(nn.Module):
    """Multi-headed attention.

    Args:
        dim: Input dimension
        n_hidden: Hidden dimension per head
        num_heads: Number of attention heads
    """

    def __init__(self, dim: int, n_hidden: int, num_heads: int):
        super().__init__()

        self.heads = nn.ModuleList(
            [AttentionHead(dim, n_hidden) for _ in range(num_heads)]
        )
        self.W_O = nn.Linear(n_hidden * num_heads, dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply multi-headed attention.

        Args:
            x: Input tensor (B, T, dim)
            attn_mask: Attention mask (B, T, T) or None

        Returns:
            attn_output: Attention output (B, T, dim)
            attn_alphas: Attention weights per head (B, num_heads, T, T)
        """
        attn_outputs = []
        attn_alphas = []

        for head in self.heads:
            attn_output, attn_alpha = head(x, attn_mask)
            attn_outputs.append(attn_output)
            attn_alphas.append(attn_alpha)

        attn_output = self.W_O(torch.cat(attn_outputs, dim=-1))
        attn_alphas = torch.stack(attn_alphas, dim=1)

        return attn_output, attn_alphas


class FFN(nn.Module):
    """Feed-forward network with layer norm.

    Args:
        dim: Input/output dimension
        n_hidden: Hidden layer width
    """

    def __init__(self, dim: int, n_hidden: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network.

        Args:
            x: Input tensor (B, T, dim)

        Returns:
            Output tensor (B, T, dim)
        """
        return self.net(x)


class AttentionResidual(nn.Module):
    """Transformer block with attention and FFN, both with residual connections.

    Args:
        dim: Input/output dimension
        attn_dim: Hidden dimension for attention
        mlp_dim: Hidden dimension for FFN
        num_heads: Number of attention heads
    """

    def __init__(self, dim: int, attn_dim: int, mlp_dim: int, num_heads: int):
        super().__init__()

        self.attn = MultiHeadedAttention(dim, attn_dim, num_heads)
        self.ffn = FFN(dim, mlp_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply attention block with residual connections.

        Args:
            x: Input tensor (B, T, dim)
            attn_mask: Attention mask (B, T, T) or None

        Returns:
            output: Output tensor (B, T, dim)
            alphas: Attention weights (B, num_heads, T, T)
        """
        attn_out, alphas = self.attn(x=x, attn_mask=attn_mask)
        x = attn_out + x  # Residual
        x = self.ffn(x) + x  # Residual

        return x, alphas


class Transformer(nn.Module):
    """Transformer encoder with multiple attention layers.

    Args:
        dim: Input/output dimension
        attn_dim: Hidden dimension for attention
        mlp_dim: Hidden dimension for FFN
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
    """

    def __init__(
        self,
        dim: int,
        attn_dim: int,
        mlp_dim: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                AttentionResidual(dim, attn_dim, mlp_dim, num_heads)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply transformer encoder.

        Args:
            x: Input tensor (B, T, dim)
            attn_mask: Attention mask (B, T, T) or None
            return_attn: Whether to return attention weights

        Returns:
            output: Output tensor (B, T, dim)
            collected_attns: Attention weights (B, num_layers, num_heads, T, T)
                            or None if return_attn=False
        """
        collected_attns = [] if return_attn else None

        for layer in self.layers:
            x, alpha = layer(x, attn_mask)
            if return_attn:
                collected_attns.append(alpha)

        if return_attn:
            collected_attns = torch.stack(collected_attns, dim=1)

        return x, collected_attns
