#!/usr/bin/env python3
"""Training entry point for LOAF weather forecasting models.

Trains an MPNN (graph) or VisionTransformer (attention) model to predict
wind (u, v) at multiple forecast horizons, from downloaded IEM station data
and (optionally) HRRR/ERA5 grid data. See software/config/*.yaml for region
configs.

Usage (from the repo root):
    python software/scripts/train.py --config software/config/arlington.yaml
    python software/scripts/train.py --config software/config/seattle.yaml \\
        --model-type vit --epochs 20
"""

import argparse
import logging
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from loaf.config import load_config  # noqa: E402
from loaf.data.loaders import create_dataloaders  # noqa: E402
from loaf.training import Trainer, build_model  # noqa: E402

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_year(iem_dir: Path) -> int:
    """Pick the year with the most downloaded IEM files, when --year isn't given."""
    files = list(iem_dir.glob("iem_*.parquet")) + list(iem_dir.glob("iem_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No IEM data found in {iem_dir} - download data first (see "
            f"loaf-download-iem) or pass --year explicitly."
        )

    years = Counter(int(f.stem.split("_")[1]) for f in files)
    return years.most_common(1)[0][0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a LOAF weather forecasting model (MPNN or ViT)."
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to a LOAF YAML config file (e.g. software/config/arlington.yaml).",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Base directory containing downloaded data (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for checkpoints/logs (default: runs/<region>_<timestamp>)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year of data to train on (default: year with the most downloaded IEM data)",
    )
    parser.add_argument(
        "--model-type", choices=["mpnn", "vit"], default=None, help="Override config model.type"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override config training.epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override config training.batch_size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Override config training.learning_rate"
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=24,
        help="Minimum real observations required to keep a station (default: 24)",
    )
    parser.add_argument(
        "--use-hrrr",
        action="store_true",
        help="Fuse HRRR grid data (requires hourly HRRR coverage over the training window)",
    )
    parser.add_argument(
        "--use-era5",
        action="store_true",
        help="Fuse ERA5 grid data (requires hourly ERA5 coverage over the training window)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="DataLoader worker processes (default: 0)"
    )
    parser.add_argument(
        "--device", default=None, help="Device to train on (default: cuda if available, else cpu)"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = load_config(args.config)
    region_cfg = config.get("region", {})
    data_cfg = config.get("data", {})
    model_cfg = dict(config.get("model", {}))
    training_cfg = config.get("training", {})

    if args.model_type:
        model_cfg["type"] = args.model_type
    model_cfg.setdefault("type", "mpnn")

    epochs = args.epochs or training_cfg.get("epochs", 100)
    batch_size = args.batch_size or training_cfg.get("batch_size", 32)
    learning_rate = args.learning_rate or training_cfg.get("learning_rate", 1e-4)
    weight_decay = training_cfg.get("weight_decay", 1e-4)
    val_split = training_cfg.get("val_split", 0.15)
    patience = training_cfg.get("patience", 10)
    max_grad_norm = training_cfg.get("max_grad_norm", 1.0)
    seed = training_cfg.get("seed", 42)

    set_seed(seed)

    back_hrs = data_cfg.get("back_hrs", 24)
    lead_times = data_cfg.get("lead_times") or [data_cfg.get("lead_hrs", 48)]

    lat_bounds = (region_cfg.get("lat_min"), region_cfg.get("lat_max"))
    lon_bounds = (region_cfg.get("lon_min"), region_cfg.get("lon_max"))

    data_dir = Path(args.data_dir)
    year = args.year or infer_year(data_dir / "iem")

    region_name = region_cfg.get("name", "run")
    logger.info(
        f"Training {model_cfg['type']} model for region '{region_name}', year {year}, "
        f"lead_times={lead_times}h"
    )

    bundle = create_dataloaders(
        data_dir=data_dir,
        year=year,
        back_hrs=back_hrs,
        lead_times=lead_times,
        batch_size=batch_size,
        val_split=val_split,
        num_workers=args.num_workers,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        use_era5=args.use_era5,
        use_hrrr=args.use_hrrr,
        min_observations=args.min_observations,
    )

    dataset = bundle.dataset
    logger.info(
        f"Loaded {len(dataset)} samples, {dataset.n_stations} stations "
        f"({len(bundle.train_loader.dataset)} train / {len(bundle.val_loader.dataset)} val)"
    )

    model = build_model(
        model_cfg,
        n_stations=dataset.n_stations,
        in_hrs=back_hrs,
        n_station_vars=len(dataset.station_vars),
        n_out_features=len(lead_times) * len(dataset.target_vars),
        grid_vars=dataset.grid_vars if dataset.grid_loader is not None else None,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Built {model_cfg['type']} model with {n_params:,} parameters")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("runs") / f"{region_name}_{datetime.now():%Y%m%d_%H%M%S}"
    )

    target_stats = {
        var: dataset.station_stats[var]
        for var in dataset.target_vars
        if var in dataset.station_stats
    }

    trainer = Trainer(
        model=model,
        model_type=model_cfg["type"],
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        output_dir=output_dir,
        lead_times=dataset.lead_times,
        target_vars=dataset.target_vars,
        station_vars=dataset.station_vars,
        target_stats=target_stats,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        patience=patience,
        device=args.device,
    )

    state = trainer.fit(epochs)

    logger.info(
        f"Training complete. Best val loss {state.best_val_loss:.4f} at epoch "
        f"{state.best_epoch}. Checkpoints saved to {output_dir}"
    )


if __name__ == "__main__":
    main()
