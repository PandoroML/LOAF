"""Normalization utilities for weather data.

Provides normalizers that can encode data to a normalized range
and decode back to original values.

Adapted from LocalizedWeather Normalizers.py.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class NormalizationStats:
    """Container for normalization statistics."""

    min: float
    max: float
    mean: float
    std: float


class MinMaxNormalizer:
    """Min-max normalization to [0, 1] range.

    Args:
        min_val: Minimum value for normalization.
        max_val: Maximum value for normalization.
        eps: Small value to prevent division by zero.
    """

    def __init__(self, min_val: float, max_val: float, eps: float = 1e-5):
        self.min = min_val
        self.max = max_val
        self.eps = eps

    def encode(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Normalize to [0, 1]."""
        return (x - self.min) / (self.max - self.min + self.eps)

    def decode(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Denormalize from [0, 1]."""
        return x * (self.max - self.min + self.eps) + self.min

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for serialization."""
        return {"min": self.min, "max": self.max, "eps": self.eps}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "MinMaxNormalizer":
        """Create from dictionary."""
        return cls(d["min"], d["max"], d.get("eps", 1e-5))


class RangeNormalizer:
    """Normalize to arbitrary [a, b] range.

    Args:
        min_val: Minimum value in the data.
        max_val: Maximum value in the data.
        a: Target range minimum.
        b: Target range maximum.
        eps: Small value to prevent division by zero.
    """

    def __init__(
        self,
        min_val: float,
        max_val: float,
        a: float = -1.0,
        b: float = 1.0,
        eps: float = 1e-5,
    ):
        self.min = min_val
        self.max = max_val
        self.a = a
        self.b = b
        self.eps = eps
        self.delta = (self.b - self.a) / (self.max - self.min + self.eps)

    def encode(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Normalize to [a, b]."""
        return (x - self.min) * self.delta + self.a

    def decode(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Denormalize from [a, b]."""
        return (x - self.a) / self.delta + self.min

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "min": self.min,
            "max": self.max,
            "a": self.a,
            "b": self.b,
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "RangeNormalizer":
        """Create from dictionary."""
        return cls(d["min"], d["max"], d["a"], d["b"], d.get("eps", 1e-5))


class StandardNormalizer:
    """Standard (z-score) normalization.

    Normalizes data to zero mean and unit variance.

    Args:
        mean: Mean value for normalization.
        std: Standard deviation for normalization.
        eps: Small value to prevent division by zero.
    """

    def __init__(self, mean: float, std: float, eps: float = 1e-5):
        self.mean = mean
        self.std = std
        self.eps = eps

    def encode(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Normalize to zero mean, unit variance."""
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Denormalize from standard normal."""
        return x * (self.std + self.eps) + self.mean

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for serialization."""
        return {"mean": self.mean, "std": self.std, "eps": self.eps}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "StandardNormalizer":
        """Create from dictionary."""
        return cls(d["mean"], d["std"], d.get("eps", 1e-5))


# Type alias for any normalizer
Normalizer = MinMaxNormalizer | RangeNormalizer | StandardNormalizer


class NormalizerCollection:
    """Collection of normalizers for multiple variables.

    Provides a convenient way to manage normalizers for all variables
    in a dataset.

    Args:
        normalizer_type: Type of normalizer to use ("minmax", "standard", "range").
        range_bounds: Tuple of (a, b) for range normalization.
    """

    def __init__(
        self,
        normalizer_type: str = "minmax",
        range_bounds: tuple[float, float] | None = None,
    ):
        self.normalizer_type = normalizer_type
        self.range_bounds = range_bounds or (-1.0, 1.0)
        self._normalizers: dict[str, Normalizer] = {}

    def fit(
        self,
        data: dict[str, np.ndarray | torch.Tensor],
    ) -> "NormalizerCollection":
        """Fit normalizers to data.

        Args:
            data: Dictionary mapping variable names to data arrays.

        Returns:
            Self for chaining.
        """
        for var, values in data.items():
            if isinstance(values, torch.Tensor):
                values = values.numpy()

            stats = {
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
            }

            self._normalizers[var] = self._create_normalizer(stats)

        return self

    def fit_from_stats(
        self,
        stats: dict[str, dict[str, float]],
    ) -> "NormalizerCollection":
        """Fit normalizers from precomputed statistics.

        Args:
            stats: Dictionary mapping variable names to stat dictionaries
                with keys 'min', 'max', 'mean', 'std'.

        Returns:
            Self for chaining.
        """
        for var, var_stats in stats.items():
            self._normalizers[var] = self._create_normalizer(var_stats)
        return self

    def _create_normalizer(self, stats: dict[str, float]) -> Normalizer:
        """Create a normalizer from statistics."""
        if self.normalizer_type == "minmax":
            return MinMaxNormalizer(stats["min"], stats["max"])
        elif self.normalizer_type == "standard":
            return StandardNormalizer(stats["mean"], stats["std"])
        elif self.normalizer_type == "range":
            return RangeNormalizer(
                stats["min"],
                stats["max"],
                self.range_bounds[0],
                self.range_bounds[1],
            )
        else:
            raise ValueError(f"Unknown normalizer type: {self.normalizer_type}")

    def encode(
        self,
        var: str,
        x: torch.Tensor | np.ndarray,
    ) -> torch.Tensor | np.ndarray:
        """Encode a variable."""
        if var not in self._normalizers:
            return x
        return self._normalizers[var].encode(x)

    def decode(
        self,
        var: str,
        x: torch.Tensor | np.ndarray,
    ) -> torch.Tensor | np.ndarray:
        """Decode a variable."""
        if var not in self._normalizers:
            return x
        return self._normalizers[var].decode(x)

    def __contains__(self, var: str) -> bool:
        """Check if normalizer exists for variable."""
        return var in self._normalizers

    def __getitem__(self, var: str) -> Normalizer:
        """Get normalizer for variable."""
        return self._normalizers[var]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "normalizer_type": self.normalizer_type,
            "range_bounds": self.range_bounds,
            "normalizers": {
                var: norm.to_dict() for var, norm in self._normalizers.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NormalizerCollection":
        """Create from dictionary."""
        collection = cls(d["normalizer_type"], tuple(d.get("range_bounds", (-1, 1))))

        # Recreate normalizers
        normalizer_cls = {
            "minmax": MinMaxNormalizer,
            "standard": StandardNormalizer,
            "range": RangeNormalizer,
        }[d["normalizer_type"]]

        for var, norm_dict in d["normalizers"].items():
            collection._normalizers[var] = normalizer_cls.from_dict(norm_dict)

        return collection

    def save(self, path: str) -> None:
        """Save normalizers to JSON file."""
        import json

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "NormalizerCollection":
        """Load normalizers from JSON file."""
        import json

        with open(path) as f:
            return cls.from_dict(json.load(f))
