"""Training pipeline modules.

This module provides the training infrastructure for LOAF models:
- Trainer: Main training loop with checkpointing and early stopping
- TrainingConfig: Configuration dataclass for training parameters
- Loss functions: MSE, WindSpeed, MAE, Combined
- Metrics: Evaluation metrics for weather forecasting
"""

from loaf.training.evaluate import (
    CombinedLoss,
    MAELoss,
    Metrics,
    MSELoss,
    WindSpeedLoss,
    get_loss_function,
    wind_speed_error,
)
from loaf.training.trainer import (
    Trainer,
    TrainingConfig,
    TrainingState,
)

__all__ = [
    # Trainer
    "Trainer",
    "TrainingConfig",
    "TrainingState",
    # Loss functions
    "MSELoss",
    "MAELoss",
    "WindSpeedLoss",
    "CombinedLoss",
    "get_loss_function",
    # Metrics
    "Metrics",
    "wind_speed_error",
]
