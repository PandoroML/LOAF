#!/usr/bin/env python3
"""Training script for LOAF weather forecasting models.

Usage:
    python scripts/train.py --config config/seattle.yaml
    python scripts/train.py --config config/seattle.yaml --model vit --epochs 50
    python scripts/train.py --config config/seattle.yaml --checkpoint checkpoints/
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add software directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from loaf.config import get_config
from loaf.data.loaders.dataset import WeatherDataset, create_dataloaders
from loaf.data.loaders.iem import IEMLoader
from loaf.data.loaders.hrrr import HRRRLoader
from loaf.data.loaders.stations import StationMetadata
from loaf.model import MPNN, VisionTransformer
from loaf.training.trainer import Trainer, TrainingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train LOAF weather forecasting model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="config/seattle.yaml",
        help="Path to configuration YAML file",
    )

    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Year of data to use for training",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        choices=["gnn", "vit"],
        default="gnn",
        help="Model type to train",
    )

    # Training overrides
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (overrides config)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint directory for saving models",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu, mps)",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def create_model(
    model_type: str,
    config,
    n_stations: int,
    back_hrs: int,
    n_vars_in: int = 4,
    n_vars_out: int = 2,
    n_grid: int | None = None,
):
    """Create model based on type and configuration.

    Args:
        model_type: "gnn" or "vit".
        config: Configuration object.
        n_stations: Number of weather stations.
        back_hrs: Number of historical hours.
        n_vars_in: Number of input variables.
        n_vars_out: Number of output variables.
        n_grid: Number of grid points (for GNN external layers).

    Returns:
        PyTorch model.
    """
    hidden_dim = getattr(config.model, "hidden_dim", 128)
    num_gnn_layers = getattr(config.model, "num_gnn_layers", 2)
    num_transformer_layers = getattr(config.model, "num_transformer_layers", 5)
    num_heads = getattr(config.model, "num_heads", 3)

    # Calculate feature dimensions
    n_node_features_m = back_hrs * n_vars_in
    n_node_features_e = back_hrs * n_vars_in if n_grid else 0

    if model_type == "gnn":
        model = MPNN(
            n_passing=num_gnn_layers,
            lead_hrs=getattr(config.data, "lead_hrs", 48),
            n_node_features_m=n_node_features_m,
            n_node_features_e=n_node_features_e,
            n_out_features=n_vars_out,
            hidden_dim=hidden_dim,
        )
    elif model_type == "vit":
        model = VisionTransformer(
            n_stations=n_stations,
            madis_len=back_hrs,
            madis_n_vars_i=n_vars_in,
            madis_n_vars_o=n_vars_out,
            dim=hidden_dim,
            attn_dim=hidden_dim // 2,
            mlp_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Log model info
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Created {model_type.upper()} model with {n_params:,} parameters")

    return model


def main():
    """Main training function."""
    args = parse_args()

    # Set debug logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("LOAF Weather Forecasting Model Training")
    logger.info("=" * 60)

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to software directory
        config_path = Path(__file__).parent.parent / args.config
    if not config_path.exists():
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)

    config = get_config(config_path)
    logger.info(f"Loaded configuration from: {config_path}")

    # Get data directory
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = Path(__file__).parent.parent / data_dir

    # Get training parameters (CLI overrides config)
    epochs = args.epochs or getattr(config.training, "epochs", 100)
    batch_size = args.batch_size or getattr(config.training, "batch_size", 64)
    learning_rate = args.lr or getattr(config.training, "learning_rate", 1e-4)
    patience = args.patience or getattr(config.training, "patience", 10)
    val_split = getattr(config.training, "val_split", 0.15)

    # Get data parameters
    back_hrs = getattr(config.data, "back_hrs", 24)
    lead_hrs = getattr(config.data, "lead_hrs", 48)

    # Get region bounds
    lat_bounds = (config.region.lat_min, config.region.lat_max)
    lon_bounds = (config.region.lon_min, config.region.lon_max)

    logger.info(f"Region: {config.region.name}")
    logger.info(f"  Lat: {lat_bounds[0]:.2f} to {lat_bounds[1]:.2f}")
    logger.info(f"  Lon: {lon_bounds[0]:.2f} to {lon_bounds[1]:.2f}")
    logger.info(f"Training parameters:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Early stopping patience: {patience}")
    logger.info(f"  Historical window: {back_hrs} hours")
    logger.info(f"  Forecast horizon: {lead_hrs} hours")

    # Create dataloaders
    logger.info("Loading data...")
    try:
        train_loader, val_loader = create_dataloaders(
            data_dir=data_dir,
            year=args.year,
            back_hrs=back_hrs,
            lead_hours=lead_hrs,
            batch_size=batch_size,
            val_split=val_split,
            lat_bounds=lat_bounds,
            lon_bounds=lon_bounds,
            use_hrrr=True,
            use_era5=False,
        )
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
    except FileNotFoundError as e:
        logger.error(f"Data not found: {e}")
        logger.error(
            "Please ensure you have downloaded the required data using the download scripts."
        )
        sys.exit(1)

    # Get dataset info from the first batch
    sample_batch = next(iter(train_loader))
    n_stations = sample_batch["station_lon"].shape[1]
    n_grid = sample_batch.get("grid_lon", torch.zeros(1, 0)).shape[1]
    logger.info(f"Number of stations: {n_stations}")
    if n_grid > 0:
        logger.info(f"Number of grid points: {n_grid}")

    # Create model
    model = create_model(
        model_type=args.model,
        config=config,
        n_stations=n_stations,
        back_hrs=back_hrs,
        n_vars_in=4,  # u, v, temp, dewpoint
        n_vars_out=2,  # u, v
        n_grid=n_grid if n_grid > 0 else None,
    )

    # Create trainer config
    checkpoint_dir = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    training_config = TrainingConfig(
        back_hrs=back_hrs,
        learning_rate=learning_rate,
        weight_decay=getattr(config.training, "weight_decay", 1e-4),
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        max_grad_norm=getattr(config.training, "max_grad_norm", 1.0),
        checkpoint_dir=checkpoint_dir,
        save_best_only=True,
        loss_type="mse",
        output_vars=["u", "v"],
        device=args.device,
        seed=args.seed,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        config=training_config,
        model_type=args.model,
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader)

    # Final evaluation
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Best validation loss: {trainer.state.best_val_loss:.6f}")

    if history["val_metrics"]:
        final_metrics = history["val_metrics"][-1]
        for name, value in final_metrics.items():
            logger.info(f"  {name}: {value:.4f}")

    if checkpoint_dir:
        logger.info(f"Checkpoints saved to: {checkpoint_dir}")

    return history


if __name__ == "__main__":
    main()
