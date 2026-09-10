#!/usr/bin/env python3
"""One-shot LOAF pipeline: download data, train a model, and serve forecasts.

Chains what loaf-download-iem[/hrrr/era5], loaf-train, and loaf-serve each do
into a single command - the common case of "I have a region config, give me
a running forecast API." Each stage can be skipped independently to resume a
partial run.

Usage (from the repo root):
    # Full run: download 3 months, train, and serve
    loaf-pipeline --config software/config/arlington.yaml \\
        --start-date 2024-10-01 --end-date 2024-12-31 --year 2024

    # Reuse already-downloaded data, just train + serve
    loaf-pipeline --config software/config/arlington.yaml --skip-download --year 2024

    # Download + train only, don't start the server
    loaf-pipeline --config software/config/arlington.yaml \\
        --start-date 2024-10-01 --end-date 2024-12-31 --no-serve

    # Reuse an already-trained checkpoint, just serve it
    loaf-pipeline --config software/config/arlington.yaml \\
        --skip-download --skip-train --checkpoint runs/arlington_.../best.pt
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loaf.inference.predictor import Predictor  # noqa: E402
from loaf.inference.server import run_server  # noqa: E402
from loaf.pipeline import download_stage, train_stage  # noqa: E402

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download data, train a model, and serve forecasts - end to end."
    )
    parser.add_argument(
        "--config", "-c", required=True, help="Path to a LOAF YAML config file."
    )
    parser.add_argument(
        "--data-dir", default="data", help="Base directory for downloaded data (default: data)"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for training/inference (default: cuda if available, else cpu)",
    )

    download = parser.add_argument_group("1. download")
    download.add_argument(
        "--skip-download", action="store_true", help="Reuse data already on disk"
    )
    download.add_argument(
        "--start-date", help="Start date to download, YYYY-MM-DD (required unless --skip-download)"
    )
    download.add_argument(
        "--end-date", help="End date to download, YYYY-MM-DD (required unless --skip-download)"
    )
    download.add_argument(
        "--stations",
        nargs="+",
        default=None,
        help="IEM station IDs (default: config's list, or auto-discovered from region bounds)",
    )
    download.add_argument(
        "--rate-limit-delay",
        type=float,
        default=1.0,
        help="Seconds to wait between IEM per-station requests (default: 1.0)",
    )

    grid = parser.add_argument_group("grid fusion (applies to download, train, and serve)")
    grid.add_argument(
        "--use-hrrr", action="store_true", help="Download (if not skipped) and fuse HRRR grid data"
    )
    grid.add_argument(
        "--use-era5", action="store_true", help="Download (if not skipped) and fuse ERA5 grid data"
    )

    train = parser.add_argument_group("2. train")
    train.add_argument(
        "--skip-train", action="store_true", help="Skip training; requires --checkpoint"
    )
    train.add_argument(
        "--checkpoint",
        default=None,
        help="Existing checkpoint to serve when --skip-train is set",
    )
    train.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year of data to train/serve on (default: year with the most downloaded IEM data)",
    )
    train.add_argument(
        "--output-dir",
        default=None,
        help="Directory for checkpoints/logs (default: runs/<region>_<timestamp>)",
    )
    train.add_argument(
        "--model-type", choices=["mpnn", "vit"], default=None, help="Override config model.type"
    )
    train.add_argument("--epochs", type=int, default=None, help="Override config training.epochs")
    train.add_argument(
        "--batch-size", type=int, default=None, help="Override config training.batch_size"
    )
    train.add_argument(
        "--learning-rate", type=float, default=None, help="Override config training.learning_rate"
    )
    train.add_argument(
        "--min-observations",
        type=int,
        default=24,
        help="Minimum real observations required to keep a station (default: 24)",
    )
    train.add_argument(
        "--num-workers", type=int, default=0, help="DataLoader worker processes (default: 0)"
    )
    train.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating the HTML training report (report.html in the checkpoint's "
        "directory) that's otherwise built after every run.",
    )

    serve = parser.add_argument_group("3. serve")
    serve.add_argument(
        "--no-serve", action="store_true", help="Stop after training; don't start the API server"
    )
    serve.add_argument(
        "--cache-ttl",
        type=float,
        default=300.0,
        help="Seconds to reuse a forward pass before recomputing it (default: 300)",
    )
    serve.add_argument("--host", default="0.0.0.0", help="Interface to bind (default: 0.0.0.0)")
    serve.add_argument("--port", type=int, default=5000, help="Port to bind (default: 5000)")
    serve.add_argument("--debug", action="store_true", help="Run Flask in debug/reload mode")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.skip_download and (not args.start_date or not args.end_date):
        parser.error("--start-date and --end-date are required unless --skip-download is set")
    if args.skip_train and not args.checkpoint:
        parser.error("--checkpoint is required when --skip-train is set")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.skip_download:
        logger.info("[1/3] Skipping download (--skip-download)")
    else:
        logger.info("[1/3] Downloading data")
        download_stage(
            config_path=args.config,
            data_dir=args.data_dir,
            start_date=datetime.strptime(args.start_date, "%Y-%m-%d"),
            end_date=datetime.strptime(args.end_date, "%Y-%m-%d"),
            use_hrrr=args.use_hrrr,
            use_era5=args.use_era5,
            stations=args.stations,
            rate_limit_delay=args.rate_limit_delay,
        )

    if args.skip_train:
        logger.info(f"[2/3] Skipping training (--skip-train); using checkpoint {args.checkpoint}")
        checkpoint_path = Path(args.checkpoint)
    else:
        logger.info("[2/3] Training")
        checkpoint_path, _state = train_stage(
            config_path=args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            year=args.year,
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            min_observations=args.min_observations,
            use_hrrr=args.use_hrrr,
            use_era5=args.use_era5,
            num_workers=args.num_workers,
            device=args.device,
        )

    if args.no_report:
        logger.info("Skipping report generation (--no-report)")
    else:
        from loaf.reporting import build_report_data, render_html

        logger.info("Generating training report")
        report_data = build_report_data(checkpoint_path.parent, data_dir=args.data_dir)
        if report_data.inference_error:
            logger.warning(
                f"Validation inference failed, report will skip that detail: "
                f"{report_data.inference_error}"
            )
        report_path = render_html(report_data, checkpoint_path.parent / "report.html")
        logger.info(f"Report written to {report_path}")

    if args.no_serve:
        logger.info(f"[3/3] Skipping serve (--no-serve). Checkpoint: {checkpoint_path}")
        return

    logger.info(f"[3/3] Serving forecasts from {checkpoint_path}")
    predictor = Predictor(
        checkpoint_path=checkpoint_path,
        data_dir=args.data_dir,
        year=args.year,
        cache_ttl=args.cache_ttl,
        device=args.device,
    )
    logger.info(
        f"Loaded {predictor.model_type} model - {predictor.n_stations} stations, "
        f"lead_times={predictor.lead_times}h, target_vars={predictor.target_vars}, "
        f"year={predictor.year}"
    )
    run_server(predictor, host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
