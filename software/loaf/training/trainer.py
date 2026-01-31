"""Training loop for weather forecasting models.

Adapted from LocalizedWeather: Main.py and EvaluateModel.py
Authors: Qidong Yang & Jonathan Giezendanner (original)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from loaf.training.evaluate import Metrics, get_loss_function

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    batch_size: int = 64

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-6

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Learning rate scheduling
    lr_scheduler: str = "reduce_on_plateau"
    lr_factor: float = 0.5
    lr_patience: int = 5

    # Checkpointing
    checkpoint_dir: Path | None = None
    save_best_only: bool = True
    save_every: int = 10

    # Logging
    log_interval: int = 10

    # Loss function
    loss_type: str = "mse"

    # Output variables
    output_vars: list[str] = field(default_factory=lambda: ["u", "v"])

    # Device
    device: str = "auto"

    # Random seed
    seed: int = 42


@dataclass
class TrainingState:
    """Training state for checkpointing and resumption."""

    epoch: int = 0
    best_val_loss: float = float("inf")
    epochs_without_improvement: int = 0
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)


class Trainer:
    """Trainer for weather forecasting models.

    Supports both GNN (MPNN) and Transformer (ViT) models.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig | None = None,
        model_type: str = "gnn",
    ):
        """Initialize trainer.

        Args:
            model: PyTorch model to train.
            config: Training configuration.
            model_type: Type of model ("gnn" or "vit").
        """
        self.config = config or TrainingConfig()
        self.model_type = model_type

        # Set device
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.config.device)

        logger.info(f"Using device: {self.device}")

        # Move model to device
        self.model = model.to(self.device)

        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Initialize scheduler
        if self.config.lr_scheduler == "reduce_on_plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.lr_factor,
                patience=self.config.lr_patience,
            )
        else:
            self.scheduler = None

        # Initialize loss function
        self.loss_fn = get_loss_function(
            self.config.loss_type,
            self.config.output_vars,
        )

        # Initialize metrics
        self.metrics = Metrics(self.config.output_vars)

        # Training state
        self.state = TrainingState()

        # Set random seed
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> dict[str, Any]:
        """Run training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader (optional).

        Returns:
            Dictionary with training history and final metrics.
        """
        logger.info("Starting training...")
        logger.info(f"  Epochs: {self.config.epochs}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info(f"  Device: {self.device}")

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
        }

        for epoch in range(self.state.epoch, self.config.epochs):
            self.state.epoch = epoch

            # Training epoch
            train_loss, train_metrics = self._train_epoch(train_loader, epoch)
            history["train_loss"].append(train_loss)
            history["train_metrics"].append(train_metrics)
            self.state.train_losses.append(train_loss)

            # Validation epoch
            if val_loader is not None:
                val_loss, val_metrics = self._validate_epoch(val_loader, epoch)
                history["val_loss"].append(val_loss)
                history["val_metrics"].append(val_metrics)
                self.state.val_losses.append(val_loss)

                # Learning rate scheduling
                if self.scheduler is not None:
                    self.scheduler.step(val_loss)

                # Early stopping check
                if val_loss < self.state.best_val_loss - self.config.min_delta:
                    self.state.best_val_loss = val_loss
                    self.state.epochs_without_improvement = 0

                    # Save best model
                    if self.config.checkpoint_dir and self.config.save_best_only:
                        self._save_checkpoint("best.pt")
                else:
                    self.state.epochs_without_improvement += 1
                    if self.state.epochs_without_improvement >= self.config.patience:
                        logger.info(
                            f"Early stopping at epoch {epoch + 1} "
                            f"(no improvement for {self.config.patience} epochs)"
                        )
                        break

            # Periodic checkpointing
            if (
                self.config.checkpoint_dir
                and not self.config.save_best_only
                and (epoch + 1) % self.config.save_every == 0
            ):
                self._save_checkpoint(f"epoch_{epoch + 1}.pt")

            # Log epoch summary
            self._log_epoch_summary(epoch, train_loss, train_metrics, val_loader is not None)

        # Save final checkpoint
        if self.config.checkpoint_dir:
            self._save_checkpoint("final.pt")

        logger.info("Training complete!")
        logger.info(f"Best validation loss: {self.state.best_val_loss:.6f}")

        return history

    def _train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> tuple[float, dict[str, float]]:
        """Run a single training epoch.

        Args:
            loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Tuple of (loss, metrics dictionary).
        """
        self.model.train()
        self.metrics.reset()

        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1} [Train]", leave=False)

        for batch_idx, batch in enumerate(pbar):
            loss, output, target = self._train_step(batch)

            total_loss += loss.item()
            n_batches += 1

            # Update metrics
            self.metrics.update(output, target)

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Log periodically
            if (batch_idx + 1) % self.config.log_interval == 0:
                avg_loss = total_loss / n_batches
                logger.debug(
                    f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(loader)}, "
                    f"Loss: {avg_loss:.4f}"
                )

        avg_loss = total_loss / n_batches
        metrics = self.metrics.compute()

        return avg_loss, metrics

    def _train_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a single training step.

        Args:
            batch: Batch of data from dataloader.

        Returns:
            Tuple of (loss, predictions, targets).
        """
        self.optimizer.zero_grad()

        # Forward pass
        output, target = self._forward(batch)

        # Compute loss
        loss = self.loss_fn(output, target)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            if isinstance(self.model, nn.DataParallel):
                torch.nn.utils.clip_grad_norm_(
                    self.model.module.parameters(),
                    self.config.max_grad_norm,
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

        # Optimizer step
        self.optimizer.step()

        return loss, output.detach(), target.detach()

    def _validate_epoch(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> tuple[float, dict[str, float]]:
        """Run a single validation epoch.

        Args:
            loader: Validation data loader.
            epoch: Current epoch number.

        Returns:
            Tuple of (loss, metrics dictionary).
        """
        self.model.eval()
        self.metrics.reset()

        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1} [Val]", leave=False)

        with torch.no_grad():
            for batch in pbar:
                output, target = self._forward(batch)
                loss = self.loss_fn(output, target)

                total_loss += loss.item()
                n_batches += 1

                # Update metrics
                self.metrics.update(output, target)

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / n_batches
        metrics = self.metrics.compute()

        return avg_loss, metrics

    def _forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass through the model.

        Args:
            batch: Batch of data from dataloader.

        Returns:
            Tuple of (predictions, targets).
        """
        # Extract data from batch based on model type
        if self.model_type == "gnn":
            return self._forward_gnn(batch)
        elif self.model_type == "vit":
            return self._forward_vit(batch)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _forward_gnn(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for GNN (MPNN) model.

        Args:
            batch: Batch of data from dataloader.

        Returns:
            Tuple of (predictions, targets).
        """
        # Get station data
        # Station observations: (batch, time, n_stations, n_vars) -> need to reshape
        # The dataset returns: {var: (batch, n_stations, time)} for each var

        # Stack input variables
        input_vars = []
        for var in ["u", "v", "temp", "dewpoint"]:
            if var in batch:
                input_vars.append(batch[var].to(self.device))

        if input_vars:
            # Stack along last dimension: (batch, n_stations, time, n_vars)
            madis_x = torch.stack(input_vars, dim=-1)
            # Transpose to match MPNN expected format: (batch, n_stations, time, n_vars)
            # Actually the dataset returns (batch, time, n_stations) per var
            # So after stacking: (batch, time, n_stations, n_vars)
            # MPNN expects: (batch, n_stations, time, n_vars)
            if madis_x.dim() == 4:
                madis_x = madis_x.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError("No station input variables found in batch")

        # Get station positions
        madis_lon = batch["station_lon"].to(self.device)
        madis_lat = batch["station_lat"].to(self.device)

        # Add extra dimension for positions if needed
        if madis_lon.dim() == 2:
            madis_lon = madis_lon.unsqueeze(-1)
            madis_lat = madis_lat.unsqueeze(-1)

        # Get station graph edges
        edge_index = batch["k_edge_index"].to(self.device)

        # Get external grid data if available
        ext_vars = []
        for var in ["ext_u", "ext_v", "ext_temp", "ext_dewpoint"]:
            if var in batch:
                ext_vars.append(batch[var].to(self.device))

        if ext_vars:
            ext_x = torch.stack(ext_vars, dim=-1)
            # Same permutation as station data
            if ext_x.dim() == 4:
                ext_x = ext_x.permute(0, 2, 1, 3).contiguous()

            ext_lon = batch["grid_lon"].to(self.device)
            ext_lat = batch["grid_lat"].to(self.device)

            if ext_lon.dim() == 2:
                ext_lon = ext_lon.unsqueeze(-1)
                ext_lat = ext_lat.unsqueeze(-1)

            edge_index_e2m = batch["ex2m_edge_index"].to(self.device)
        else:
            ext_x = None
            ext_lon = None
            ext_lat = None
            edge_index_e2m = None

        # Forward pass
        output = self.model(
            madis_x,
            madis_lon,
            madis_lat,
            edge_index,
            ext_lon,
            ext_lat,
            ext_x,
            edge_index_e2m,
        )
        # output: (batch, n_stations, n_out_vars)

        # Get target (last timestep of input vars that are in output_vars)
        target_vars = []
        for var in self.config.output_vars:
            if var in batch:
                # Get last timestep
                target_vars.append(batch[var][:, -1, :].to(self.device))

        target = torch.stack(target_vars, dim=-1)
        # target: (batch, n_stations, n_out_vars)

        return output, target

    def _forward_vit(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for ViT model.

        Args:
            batch: Batch of data from dataloader.

        Returns:
            Tuple of (predictions, targets).
        """
        # Stack input variables
        input_vars = []
        for var in ["u", "v", "temp", "dewpoint"]:
            if var in batch:
                input_vars.append(batch[var].to(self.device))

        if input_vars:
            # Stack along last dimension: (batch, time, n_stations, n_vars)
            madis_x = torch.stack(input_vars, dim=-1)
            # ViT expects: (batch, n_stations, time, n_vars)
            if madis_x.dim() == 4:
                madis_x = madis_x.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError("No station input variables found in batch")

        # Get ERA5 data if available
        era5_vars = []
        for var in ["ext_u", "ext_v", "ext_temp", "ext_dewpoint"]:
            if var in batch:
                era5_vars.append(batch[var].to(self.device))

        if era5_vars:
            era5_x = torch.stack(era5_vars, dim=-1)
            if era5_x.dim() == 4:
                era5_x = era5_x.permute(0, 2, 1, 3).contiguous()
        else:
            era5_x = None

        # Forward pass
        output, _ = self.model(madis_x, era5_x, return_attn=False)
        # output: (batch, n_stations, n_out_vars)

        # Get target (last timestep)
        target_vars = []
        for var in self.config.output_vars:
            if var in batch:
                target_vars.append(batch[var][:, -1, :].to(self.device))

        target = torch.stack(target_vars, dim=-1)

        return output, target

    def _log_epoch_summary(
        self,
        epoch: int,
        train_loss: float,
        train_metrics: dict[str, float],
        has_val: bool,
    ) -> None:
        """Log epoch summary."""
        msg = f"Epoch {epoch + 1}/{self.config.epochs}"
        msg += f" | Train Loss: {train_loss:.4f}"

        if "wind_speed_error" in train_metrics:
            msg += f" | Wind Err: {train_metrics['wind_speed_error']:.4f}"

        if has_val and self.state.val_losses:
            msg += f" | Val Loss: {self.state.val_losses[-1]:.4f}"

        msg += f" | LR: {self.optimizer.param_groups[0]['lr']:.2e}"

        logger.info(msg)

    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename.
        """
        if self.config.checkpoint_dir is None:
            return

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": self.state.epoch,
            "model_state_dict": (
                self.model.module.state_dict()
                if isinstance(self.model, nn.DataParallel)
                else self.model.state_dict()
            ),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "state": {
                "epoch": self.state.epoch,
                "best_val_loss": self.state.best_val_loss,
                "epochs_without_improvement": self.state.epochs_without_improvement,
                "train_losses": self.state.train_losses,
                "val_losses": self.state.val_losses,
            },
            "config": self.config,
        }

        path = checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)

        # Load model state
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load training state
        state_dict = checkpoint["state"]
        self.state.epoch = state_dict["epoch"]
        self.state.best_val_loss = state_dict["best_val_loss"]
        self.state.epochs_without_improvement = state_dict["epochs_without_improvement"]
        self.state.train_losses = state_dict["train_losses"]
        self.state.val_losses = state_dict["val_losses"]

        logger.info(f"Loaded checkpoint from epoch {self.state.epoch}")

    def evaluate(
        self,
        loader: DataLoader,
    ) -> tuple[float, dict[str, float]]:
        """Evaluate model on a dataset.

        Args:
            loader: Data loader for evaluation.

        Returns:
            Tuple of (loss, metrics dictionary).
        """
        return self._validate_epoch(loader, 0)
