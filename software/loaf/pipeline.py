"""Programmatic stages of the LOAF pipeline: download -> train -> serve.

`scripts/train.py` and `scripts/pipeline.py` are both thin argparse wrappers
around train_stage() (and, for pipeline.py, download_stage()) here, so the
"loaf-train" and "loaf-pipeline" commands can't drift out of sync on what
"train from a config" actually means.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from loaf.config import load_config
from loaf.data.loaders import create_dataloaders
from loaf.training import Trainer, TrainerState, build_model

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_year(iem_dir: Path) -> int:
    """Pick the year with the most downloaded IEM files, when no year is given."""
    files = list(iem_dir.glob("iem_*.parquet")) + list(iem_dir.glob("iem_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No IEM data found in {iem_dir} - download data first (see "
            f"loaf-download-iem / loaf-pipeline) or pass year explicitly."
        )

    years = Counter(int(f.stem.split("_")[1]) for f in files)
    return years.most_common(1)[0][0]


def _to_hrrr_lon(lon: float) -> float:
    """Convert a -180/180 longitude to HRRR/Herbie's 0-360 convention."""
    return lon + 360 if lon < 0 else lon


def download_stage(
    config_path: str | Path,
    data_dir: str | Path,
    start_date: datetime,
    end_date: datetime,
    use_hrrr: bool = False,
    use_era5: bool = False,
    stations: list[str] | None = None,
    rate_limit_delay: float = 1.0,
) -> None:
    """Download IEM station data (and optionally HRRR/ERA5 grids) for a date range.

    Args:
        config_path: Path to a LOAF YAML config file - supplies region bounds
            and per-source settings (stations, variables, output dirs).
        data_dir: Base directory to write iem/hrrr/era5 subdirectories into.
        start_date: First date to download.
        end_date: Last date to download (inclusive).
        use_hrrr: Also download HRRR grid data over the same range.
        use_era5: Also download ERA5 grid data over the same range (requires
            a CDS API key in ~/.cdsapirc).
        stations: IEM station IDs to download. Defaults to the config's
            station list, or auto-discovers stations within the region bounds.
        rate_limit_delay: Seconds between IEM per-station requests.
    """
    from loaf.data.download import era5 as era5_dl
    from loaf.data.download import hrrr as hrrr_dl
    from loaf.data.download import iem as iem_dl

    config = load_config(config_path)
    region_cfg = config.get("region", {})
    lat_min, lat_max = region_cfg.get("lat_min"), region_cfg.get("lat_max")
    lon_min, lon_max = region_cfg.get("lon_min"), region_cfg.get("lon_max")
    data_dir = Path(data_dir)

    iem_settings = iem_dl.load_iem_settings_from_config(str(config_path))
    stations = stations or iem_settings.get("stations")
    if not stations:
        logger.info("No IEM stations configured - discovering from region bounds")
        station_df = iem_dl.get_available_stations(lat_min, lat_max, lon_min, lon_max)
        if station_df.empty:
            raise RuntimeError("No IEM stations found in region - pass stations explicitly.")
        stations = station_df["station_id"].tolist()

    logger.info(f"Downloading IEM station data for {len(stations)} station(s): {stations}")
    iem_dl.download_iem_range(
        start_date,
        end_date,
        output_dir=data_dir / "iem",
        stations=stations,
        variables=iem_settings.get("variables") or iem_dl.DEFAULT_VARIABLES,
        format=iem_settings.get("format") or "parquet",
        request_delay=rate_limit_delay,
    )

    if use_hrrr:
        hrrr_settings = hrrr_dl.load_hrrr_settings_from_config(str(config_path))
        logger.info("Downloading HRRR grid data")
        hrrr_dl.download_hrrr_range(
            start_date,
            end_date,
            output_dir=data_dir / "hrrr",
            var_list=hrrr_settings.get("var_list") or hrrr_dl.DEFAULT_VARIABLES,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=_to_hrrr_lon(lon_min),
            lon_max=_to_hrrr_lon(lon_max),
            max_lead_hr=hrrr_settings.get("max_lead_hr") or 18,
        )

    if use_era5:
        era5_settings = era5_dl.load_era5_settings_from_config(str(config_path))
        logger.info("Downloading ERA5 grid data")
        era5_dl.download_era5_range(
            start_date.year,
            start_date.month,
            end_date.year,
            end_date.month,
            output_dir=data_dir / "era5",
            variables=era5_settings.get("variables") or era5_dl.DEFAULT_VARIABLES,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
        )


def train_stage(
    config_path: str | Path,
    data_dir: str | Path = "data",
    output_dir: str | Path | None = None,
    year: int | None = None,
    model_type: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    min_observations: int = 24,
    use_hrrr: bool = False,
    use_era5: bool = False,
    num_workers: int = 0,
    device: str | None = None,
) -> tuple[Path, TrainerState]:
    """Train a model from a LOAF YAML config against already-downloaded data.

    This is the shared implementation behind `loaf-train` and `loaf-pipeline` -
    see scripts/train.py and scripts/pipeline.py for their thin CLI wrappers.

    Returns:
        (best_checkpoint_path, final TrainerState).
    """
    config = load_config(config_path)
    region_cfg = config.get("region", {})
    data_cfg = config.get("data", {})
    model_cfg = dict(config.get("model", {}))
    training_cfg = config.get("training", {})

    if model_type:
        model_cfg["type"] = model_type
    model_cfg.setdefault("type", "mpnn")

    epochs = epochs or training_cfg.get("epochs", 100)
    batch_size = batch_size or training_cfg.get("batch_size", 32)
    learning_rate = learning_rate or training_cfg.get("learning_rate", 1e-4)
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

    data_dir = Path(data_dir)
    year = year or infer_year(data_dir / "iem")

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
        num_workers=num_workers,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        use_era5=use_era5,
        use_hrrr=use_hrrr,
        min_observations=min_observations,
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
        Path(output_dir)
        if output_dir
        else Path("runs") / f"{region_name}_{datetime.now():%Y%m%d_%H%M%S}"
    )

    target_stats = {
        var: dataset.station_stats[var]
        for var in dataset.target_vars
        if var in dataset.station_stats
    }

    # Embedded in the checkpoint so loaf.inference.Predictor can rebuild this
    # exact model and normalize live inputs without the original config file.
    run_config: dict[str, Any] = {
        "model_cfg": model_cfg,
        "back_hrs": back_hrs,
        "lat_bounds": lat_bounds,
        "lon_bounds": lon_bounds,
        "min_observations": min_observations,
        "use_hrrr": use_hrrr,
        "use_era5": use_era5,
        "grid_vars": dataset.grid_vars if dataset.grid_loader is not None else None,
        "region_name": region_name,
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
        station_stats=dataset.station_stats,
        run_config=run_config,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        patience=patience,
        device=device,
    )

    state = trainer.fit(epochs)
    logger.info(
        f"Training complete. Best val loss {state.best_val_loss:.4f} at epoch "
        f"{state.best_epoch}. Checkpoints saved to {output_dir}"
    )
    return output_dir / "best.pt", state
