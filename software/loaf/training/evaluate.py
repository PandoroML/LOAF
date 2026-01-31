"""Evaluation metrics and validation for weather forecasting.

Adapted from LocalizedWeather: EvaluateModel.py and Utils/LossFunctions.py
Authors: Qidong Yang & Jonathan Giezendanner (original)
"""

import torch
from torch import nn


def wind_speed_error(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute wind speed error from u/v components.

    This computes the magnitude of the error vector, which is the standard
    metric for wind forecast evaluation.

    Args:
        output: Predictions with u, v in first two channels (..., 2+)
        target: Targets with u, v in first two channels (..., 2+)

    Returns:
        Wind speed error for each sample.
    """
    u_error = output[..., 0] - target[..., 0]
    v_error = output[..., 1] - target[..., 1]

    # Add epsilon for numerical stability
    error = torch.sqrt(u_error**2 + v_error**2 + torch.finfo(torch.float32).eps)
    return error


class WindSpeedLoss(nn.Module):
    """Loss function based on wind speed error (magnitude of error vector)."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute wind speed loss.

        Args:
            output: Predictions (..., 2) with u, v components.
            target: Targets (..., 2) with u, v components.
            mask: Optional boolean mask for valid samples.

        Returns:
            Scalar loss value.
        """
        error = wind_speed_error(output, target)

        if mask is not None:
            error = error[mask]

        if self.reduction == "mean":
            return error.mean()
        elif self.reduction == "sum":
            return error.sum()
        else:
            return error


class MSELoss(nn.Module):
    """Standard MSE loss with optional masking."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction="none")

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute MSE loss.

        Args:
            output: Predictions.
            target: Targets.
            mask: Optional boolean mask for valid samples.

        Returns:
            Scalar loss value.
        """
        error = self.mse(output, target)

        if mask is not None:
            error = error[mask]

        if self.reduction == "mean":
            return error.mean()
        elif self.reduction == "sum":
            return error.sum()
        else:
            return error


class MAELoss(nn.Module):
    """Mean Absolute Error loss with optional masking."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute MAE loss.

        Args:
            output: Predictions.
            target: Targets.
            mask: Optional boolean mask for valid samples.

        Returns:
            Scalar loss value.
        """
        error = torch.abs(output - target)

        if mask is not None:
            error = error[mask]

        if self.reduction == "mean":
            return error.mean()
        elif self.reduction == "sum":
            return error.sum()
        else:
            return error


class CombinedLoss(nn.Module):
    """Combined loss for wind (u, v) and other variables.

    Uses wind speed error for u/v components and RMSE for other variables.
    This matches the LocalizedWeather paper methodology.
    """

    def __init__(
        self,
        output_vars: list[str],
        reduction: str = "mean",
    ):
        """Initialize combined loss.

        Args:
            output_vars: List of output variable names (e.g., ["u", "v", "temp"]).
            reduction: Reduction method ("mean" or "sum").
        """
        super().__init__()
        self.output_vars = output_vars
        self.reduction = reduction

        # Find indices for u, v, and other variables
        self.u_idx = output_vars.index("u") if "u" in output_vars else None
        self.v_idx = output_vars.index("v") if "v" in output_vars else None
        self.other_indices = [
            i for i, v in enumerate(output_vars) if v not in ["u", "v"]
        ]

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            output: Predictions (..., n_vars).
            target: Targets (..., n_vars).
            mask: Optional boolean mask for valid samples.

        Returns:
            Scalar loss value.
        """
        errors = []

        # Wind speed error for u, v components
        if self.u_idx is not None and self.v_idx is not None:
            u_error = output[..., self.u_idx] - target[..., self.u_idx]
            v_error = output[..., self.v_idx] - target[..., self.v_idx]
            wind_error = torch.sqrt(
                u_error**2 + v_error**2 + torch.finfo(torch.float32).eps
            )
            if mask is not None:
                wind_error = wind_error[mask[..., 0] if mask.dim() > 1 else mask]
            errors.append(wind_error)

        # RMSE for other variables
        for idx in self.other_indices:
            var_error = torch.sqrt(
                (output[..., idx] - target[..., idx]) ** 2
                + torch.finfo(torch.float32).eps
            )
            if mask is not None:
                var_error = var_error[mask[..., idx] if mask.dim() > 1 else mask]
            errors.append(var_error)

        if not errors:
            return torch.tensor(0.0, device=output.device)

        # Concatenate all errors
        all_errors = torch.cat([e.flatten() for e in errors])

        if self.reduction == "mean":
            return all_errors.mean()
        elif self.reduction == "sum":
            return all_errors.sum()
        else:
            return all_errors


class Metrics:
    """Compute and track evaluation metrics."""

    def __init__(self, output_vars: list[str]):
        """Initialize metrics tracker.

        Args:
            output_vars: List of output variable names.
        """
        self.output_vars = output_vars
        self.reset()

    def reset(self) -> None:
        """Reset accumulated metrics."""
        self.mse_sum = {var: 0.0 for var in self.output_vars}
        self.mae_sum = {var: 0.0 for var in self.output_vars}
        self.count = {var: 0 for var in self.output_vars}

        # Wind-specific metrics
        if "u" in self.output_vars and "v" in self.output_vars:
            self.wind_speed_error_sum = 0.0
            self.wind_count = 0

    def update(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Update metrics with a batch of predictions.

        Args:
            output: Predictions (..., n_vars).
            target: Targets (..., n_vars).
            mask: Optional boolean mask for valid samples (..., n_vars).
        """
        output = output.detach()
        target = target.detach()

        for i, var in enumerate(self.output_vars):
            pred = output[..., i]
            tgt = target[..., i]

            if mask is not None:
                var_mask = mask[..., i] if mask.dim() > 1 else mask
                pred = pred[var_mask]
                tgt = tgt[var_mask]

            n = pred.numel()
            if n > 0:
                self.mse_sum[var] += ((pred - tgt) ** 2).sum().item()
                self.mae_sum[var] += torch.abs(pred - tgt).sum().item()
                self.count[var] += n

        # Wind speed error
        if "u" in self.output_vars and "v" in self.output_vars:
            u_idx = self.output_vars.index("u")
            v_idx = self.output_vars.index("v")

            u_pred = output[..., u_idx]
            v_pred = output[..., v_idx]
            u_tgt = target[..., u_idx]
            v_tgt = target[..., v_idx]

            if mask is not None:
                u_mask = mask[..., u_idx] if mask.dim() > 1 else mask
                u_pred = u_pred[u_mask]
                v_pred = v_pred[u_mask]
                u_tgt = u_tgt[u_mask]
                v_tgt = v_tgt[u_mask]

            n = u_pred.numel()
            if n > 0:
                wind_err = torch.sqrt(
                    (u_pred - u_tgt) ** 2
                    + (v_pred - v_tgt) ** 2
                    + torch.finfo(torch.float32).eps
                )
                self.wind_speed_error_sum += wind_err.sum().item()
                self.wind_count += n

    def compute(self) -> dict[str, float]:
        """Compute final metrics.

        Returns:
            Dictionary of metric names to values.
        """
        results = {}

        # Per-variable metrics
        for var in self.output_vars:
            if self.count[var] > 0:
                results[f"mse_{var}"] = self.mse_sum[var] / self.count[var]
                results[f"rmse_{var}"] = (self.mse_sum[var] / self.count[var]) ** 0.5
                results[f"mae_{var}"] = self.mae_sum[var] / self.count[var]

        # Overall metrics
        total_mse = sum(self.mse_sum.values())
        total_count = sum(self.count.values())
        if total_count > 0:
            results["mse"] = total_mse / total_count
            results["rmse"] = (total_mse / total_count) ** 0.5

        # Wind speed error
        if hasattr(self, "wind_count") and self.wind_count > 0:
            results["wind_speed_error"] = self.wind_speed_error_sum / self.wind_count

        return results


def get_loss_function(
    loss_type: str,
    output_vars: list[str],
) -> nn.Module:
    """Get loss function by name.

    Args:
        loss_type: Loss function type ("mse", "wind_speed", "combined").
        output_vars: List of output variable names.

    Returns:
        Loss function module.
    """
    if loss_type == "mse":
        return MSELoss()
    elif loss_type == "wind_speed":
        return WindSpeedLoss()
    elif loss_type == "combined":
        return CombinedLoss(output_vars)
    elif loss_type == "mae":
        return MAELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
