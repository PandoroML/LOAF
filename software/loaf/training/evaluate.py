"""Evaluation metrics for weather forecasting models.

All metrics are computed over masked, multi-horizon predictions of shape
(n_batch, n_stations, n_lead_times, n_target_vars) so that imputed
(non-real) observations don't distort training signal or reported scores.

Adapted from LocalizedWeather EvaluateModel.py.
"""

import torch


class RunningMetrics:
    """Accumulates masked regression metrics across batches.

    Keeps running sums rather than storing every prediction, so an
    epoch's worth of validation batches can be scored with O(1) memory.

    Args:
        lead_times: Forecast horizons in hours, matching the target's
            lead-time axis (used only to label the per-horizon breakdown).
        target_vars: Names of the predicted variables, matching the
            target's variable axis (used only to label the breakdown).
    """

    def __init__(self, lead_times: list[int], target_vars: list[str]):
        self.lead_times = lead_times
        self.target_vars = target_vars
        self.reset()

    def reset(self) -> None:
        """Reset all running sums."""
        n_lead, n_vars = len(self.lead_times), len(self.target_vars)
        self._sum_sq = torch.zeros(n_lead, n_vars, dtype=torch.float64)
        self._sum_abs = torch.zeros(n_lead, n_vars, dtype=torch.float64)
        self._sum_mask = torch.zeros(n_lead, n_vars, dtype=torch.float64)
        self._sum_sq_persistence = 0.0
        self._sum_mask_persistence = 0.0

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        persistence: torch.Tensor | None = None,
    ) -> None:
        """Accumulate stats from one batch.

        Args:
            preds: Predictions (n_batch, n_stations, n_lead_times, n_target_vars).
            target: Ground truth, same shape.
            mask: 1.0 for real observations, 0.0 for imputed/missing, same shape.
            persistence: Optional naive "no change from now" baseline predictions,
                same shape, used to compute a skill score.
        """
        preds = preds.detach()
        target = target.detach()
        mask = mask.detach()

        err = (preds - target) * mask
        sq = (err ** 2).double()
        ab = err.abs().double()
        m = mask.double()

        # Sum over batch and station dims -> (n_lead_times, n_target_vars)
        self._sum_sq += sq.sum(dim=(0, 1)).cpu()
        self._sum_abs += ab.sum(dim=(0, 1)).cpu()
        self._sum_mask += m.sum(dim=(0, 1)).cpu()

        if persistence is not None:
            perr = (persistence.detach() - target) * mask
            self._sum_sq_persistence += (perr ** 2).double().sum().item()
            self._sum_mask_persistence += m.sum().item()

    def compute(self) -> dict[str, float]:
        """Compute aggregate and per-horizon/per-variable metrics.

        Returns:
            Dictionary with "loss" (mean squared error, for early stopping
            and logging), "mse", "mae", "rmse", "skill" (1 - model_mse /
            persistence_mse, NaN if no persistence baseline was given), and
            per-horizon/per-variable "rmse_{var}_{lead}h" / "mae_{var}_{lead}h".
        """
        total_mask = self._sum_mask.sum().item()
        denom = max(total_mask, 1.0)

        mse = (self._sum_sq.sum().item()) / denom
        mae = (self._sum_abs.sum().item()) / denom
        rmse = mse ** 0.5

        skill = float("nan")
        if self._sum_mask_persistence > 0:
            persistence_mse = self._sum_sq_persistence / self._sum_mask_persistence
            if persistence_mse > 0:
                skill = 1.0 - (mse / persistence_mse)

        result: dict[str, float] = {
            "loss": mse,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "skill": skill,
        }

        for i, lead_hr in enumerate(self.lead_times):
            for j, var in enumerate(self.target_vars):
                count = max(self._sum_mask[i, j].item(), 1.0)
                var_mse = self._sum_sq[i, j].item() / count
                result[f"rmse_{var}_{lead_hr}h"] = var_mse ** 0.5
                result[f"mae_{var}_{lead_hr}h"] = self._sum_abs[i, j].item() / count

        return result
