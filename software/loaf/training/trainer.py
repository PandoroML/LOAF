"""Training loop for LOAF weather forecasting models.

Supports both model backbones defined in loaf.model: the graph-based MPNN
and the attention-based VisionTransformer. Both are trained to predict
target_vars (default: u, v wind) at multiple forecast horizons in a single
forward pass, given a window of station (and optionally grid) history.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from loaf.model import MPNN, VisionTransformer
from loaf.training.evaluate import RunningMetrics

MODEL_TYPES = ("mpnn", "vit")


def build_model(
    model_cfg: dict[str, Any],
    n_stations: int,
    in_hrs: int,
    n_station_vars: int,
    n_out_features: int,
    grid_vars: list[str] | None = None,
    in_hrs_grid: int | None = None,
) -> nn.Module:
    """Build an MPNN or VisionTransformer from a model config block.

    Args:
        model_cfg: The "model" section of a LOAF YAML config (must include
            "type": "mpnn" or "vit").
        n_stations: Number of stations (needed for ViT's positional embedding).
        in_hrs: Number of historical hours in the station input window.
        n_station_vars: Number of station input variables.
        n_out_features: Total output width, i.e. n_lead_times * n_target_vars.
            The caller is responsible for reshaping the model's flat output
            back to (batch, n_stations, n_lead_times, n_target_vars).
        grid_vars: Grid variables used, if grid fusion is enabled (MPNN only).
        in_hrs_grid: Historical hours in the grid input window, if different
            from in_hrs.

    Returns:
        An uninitialized (freshly constructed) model.
    """
    model_type = model_cfg.get("type", "mpnn").lower()
    hidden_dim = model_cfg.get("hidden_dim", 128)

    if model_type == "mpnn":
        n_grid_vars = len(grid_vars) if grid_vars else 1
        n_node_features_e = (in_hrs_grid or in_hrs) * n_grid_vars
        return MPNN(
            n_passing=model_cfg.get("num_gnn_layers", 2),
            lead_hrs=0,
            n_node_features_m=in_hrs * n_station_vars,
            n_node_features_e=n_node_features_e,
            n_out_features=n_out_features,
            hidden_dim=hidden_dim,
        )
    elif model_type == "vit":
        return VisionTransformer(
            n_stations=n_stations,
            madis_len=in_hrs,
            madis_n_vars_i=n_station_vars,
            madis_n_vars_o=n_out_features,
            dim=hidden_dim,
            attn_dim=hidden_dim,
            mlp_dim=hidden_dim * 2,
            num_heads=model_cfg.get("num_heads", 3),
            num_layers=model_cfg.get("num_transformer_layers", 5),
        )
    else:
        raise ValueError(f"Unknown model.type: {model_type!r} (expected one of {MODEL_TYPES})")


@dataclass
class TrainerState:
    """Mutable training progress, returned by Trainer.fit()."""

    epoch: int = 0
    best_val_loss: float = float("inf")
    best_epoch: int = -1
    epochs_without_improvement: int = 0
    history: list[dict[str, float]] = field(default_factory=list)


class Trainer:
    """Trains an MPNN or VisionTransformer on a WeatherDataset.

    Args:
        model: Model built by build_model().
        model_type: "mpnn" or "vit" - determines the forward-pass call signature.
        train_loader: Training DataLoader (batches of WeatherDataset samples).
        val_loader: Validation DataLoader.
        output_dir: Directory for checkpoints (best.pt, last.pt) and train_log.csv.
        lead_times: Forecast horizons in hours, matching the target's lead axis.
        target_vars: Names of predicted variables, matching the target's var axis.
        station_vars: Names of station input variables, in madis_x's var axis
            order (used to build the persistence baseline for the skill metric).
        target_stats: Per-variable {"min", "max"} used to denormalize predictions
            before computing human-readable (physical-unit) metrics.
        station_stats: Per-variable {"min", "max", ...} normalization stats for
            every station input variable (a superset of target_stats). Persisted
            to the checkpoint as-is so loaf.inference.Predictor can normalize
            live inputs the same way the training data was normalized, without
            needing to recompute statistics from a short live window.
        run_config: Architecture/region metadata to embed in the checkpoint
            verbatim (e.g. model_cfg, back_hrs, lat_bounds/lon_bounds,
            min_observations, use_hrrr/use_era5, grid_vars) so a checkpoint is
            self-describing enough for loaf.inference.Predictor to rebuild the
            model and its inputs without the original training config.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        max_grad_norm: Gradient clipping norm (None to disable).
        patience: Early-stopping patience in epochs (None/0 to disable).
        device: Torch device string. Defaults to CUDA if available, else CPU.
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str | Path,
        lead_times: list[int],
        target_vars: list[str],
        station_vars: list[str],
        target_stats: dict[str, dict[str, float]] | None = None,
        station_stats: dict[str, dict[str, float]] | None = None,
        run_config: dict[str, Any] | None = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        max_grad_norm: float | None = 1.0,
        patience: int | None = 10,
        device: str | None = None,
    ):
        if model_type not in MODEL_TYPES:
            raise ValueError(f"Unknown model_type: {model_type!r} (expected one of {MODEL_TYPES})")

        self.model_type = model_type
        self.lead_times = lead_times
        self.n_lead_times = len(lead_times)
        self.target_vars = target_vars
        self.n_target_vars = len(target_vars)
        self.station_vars = station_vars
        self.target_stats = target_stats or {}
        self.station_stats = station_stats or {}
        self.run_config = run_config or {}

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.max_grad_norm = max_grad_norm
        self.patience = patience

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "train_log.csv"

        self.state = TrainerState()

    def _forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the model and reshape output to (batch, n_stations, n_lead_times, n_vars)."""
        madis_x = batch["madis_x"].to(self.device)
        ex_x = batch.get("ex_x")

        if self.model_type == "mpnn":
            madis_lon = batch["madis_lon"].to(self.device)
            madis_lat = batch["madis_lat"].to(self.device)
            edge_index = batch["edge_index"].to(self.device)

            ex_lon = ex_lat = edge_index_e2m = None
            if ex_x is not None:
                ex_x = ex_x.to(self.device)
                ex_lon = batch["ex_lon"].to(self.device)
                ex_lat = batch["ex_lat"].to(self.device)
                edge_index_e2m = batch["edge_index_e2m"].to(self.device)

            preds = self.model(
                madis_x, madis_lon, madis_lat, edge_index, ex_lon, ex_lat, ex_x, edge_index_e2m
            )
        else:  # vit
            if ex_x is not None:
                ex_x = ex_x.to(self.device)
            preds, _ = self.model(madis_x, era5_x=ex_x)

        n_batch, n_stations, _ = preds.shape
        return preds.view(n_batch, n_stations, self.n_lead_times, self.n_target_vars)

    def _compute_loss(self, preds: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Masked MSE over real (non-imputed) target observations only."""
        target = batch["target"].to(self.device)
        mask = batch["target_mask"].to(self.device)

        squared_error = (preds - target) ** 2 * mask
        denom = mask.sum().clamp_min(1.0)
        return squared_error.sum() / denom

    def _persistence_baseline(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Naive "no change from now" forecast, for the skill-score baseline."""
        try:
            indices = [self.station_vars.index(var) for var in self.target_vars]
        except ValueError:
            return None

        madis_x = batch["madis_x"].to(self.device)
        last_values = madis_x[:, :, -1, indices]  # (batch, n_stations, n_target_vars)
        return last_values.unsqueeze(2).expand(-1, -1, self.n_lead_times, -1)

    def _denormalize(self, values: torch.Tensor, var: str) -> torch.Tensor:
        """Undo min-max normalization for human-readable metrics."""
        stats = self.target_stats.get(var)
        if stats is None:
            return values
        eps = 1e-5
        return values * (stats["max"] - stats["min"] + eps) + stats["min"]

    def _denormalize_targets(self, values: torch.Tensor) -> torch.Tensor:
        """Denormalize a (..., n_target_vars) tensor, one slice per variable."""
        if not self.target_stats:
            return values
        slices = [
            self._denormalize(values[..., i], var) for i, var in enumerate(self.target_vars)
        ]
        return torch.stack(slices, dim=-1)

    def train_epoch(self) -> float:
        """Run one training epoch. Returns mean training loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad()
            preds = self._forward(batch)
            loss = self._compute_loss(preds, batch)
            loss.backward()

            if self.max_grad_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate_epoch(self) -> dict[str, float]:
        """Run one validation epoch. Returns metrics from RunningMetrics.compute()."""
        self.model.eval()
        metrics = RunningMetrics(self.lead_times, self.target_vars)

        for batch in self.val_loader:
            preds = self._forward(batch)
            target = batch["target"].to(self.device)
            mask = batch["target_mask"].to(self.device)
            persistence = self._persistence_baseline(batch)

            denorm_persistence = (
                self._denormalize_targets(persistence) if persistence is not None else None
            )
            metrics.update(
                self._denormalize_targets(preds),
                self._denormalize_targets(target),
                mask,
                persistence=denorm_persistence,
            )

        return metrics.compute()

    def fit(self, epochs: int) -> TrainerState:
        """Train for up to `epochs` epochs, with checkpointing and early stopping."""
        self._init_log()

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_metrics = self.validate_epoch()
            val_loss = val_metrics["loss"]

            print(
                f"Epoch {epoch}/{epochs} - train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_mae={val_metrics['mae']:.4f} "
                f"val_rmse={val_metrics['rmse']:.4f} val_skill={val_metrics['skill']:.4f}"
            )

            self._log_epoch(epoch, train_loss, val_metrics)
            self.state.epoch = epoch
            self.state.history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
            self._save_checkpoint("last.pt")

            if val_loss < self.state.best_val_loss:
                self.state.best_val_loss = val_loss
                self.state.best_epoch = epoch
                self.state.epochs_without_improvement = 0
                self._save_checkpoint("best.pt")
            else:
                self.state.epochs_without_improvement += 1
                if self.patience and self.state.epochs_without_improvement >= self.patience:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {self.patience} epochs)"
                    )
                    break

        return self.state

    def _init_log(self) -> None:
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["epoch", "train_loss", "val_loss", "val_mae", "val_rmse", "val_skill"]
                )

    def _log_epoch(self, epoch: int, train_loss: float, val_metrics: dict[str, float]) -> None:
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch, train_loss, val_metrics["loss"], val_metrics["mae"],
                 val_metrics["rmse"], val_metrics["skill"]]
            )

    def _save_checkpoint(self, filename: str) -> None:
        torch.save(
            {
                "epoch": self.state.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_type": self.model_type,
                "lead_times": self.lead_times,
                "target_vars": self.target_vars,
                "station_vars": self.station_vars,
                "target_stats": self.target_stats,
                "station_stats": self.station_stats,
                "run_config": self.run_config,
            },
            self.output_dir / filename,
        )
