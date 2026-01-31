"""Data handling modules for LOAF.

This module provides:
- download: Functions for downloading weather data (ERA5, HRRR, IEM)
- loaders: PyTorch data loaders for training
- preprocessing: Data normalization utilities
"""

# Import submodules - download may fail if dependencies are missing
try:
    from . import download
except ImportError:
    download = None  # type: ignore

from . import loaders, preprocessing

__all__ = ["download", "loaders", "preprocessing"]
