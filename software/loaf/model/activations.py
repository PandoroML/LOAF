"""Custom activation functions for the LOAF model.

Ported from LocalizedWeather: Modules/Activations.py
Authors: Qidong Yang & Jonathan Giezendanner (original)
"""

import torch
from torch import nn


class Tanh(nn.Module):
    """Tanh activation function as a module."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


class Swish(nn.Module):
    """Swish activation function (x * sigmoid(beta * x))."""

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)
