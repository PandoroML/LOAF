#!/usr/bin/env python3
"""Training entry point for LOAF weather forecasting models.

Trains an MPNN (graph) or VisionTransformer (attention) model to predict
wind (u, v) at multiple forecast horizons, from downloaded IEM station data
and (optionally) HRRR/ERA5 grid data. See software/config/*.yaml for region
configs.

This is a thin CLI wrapper around loaf.pipeline.train_stage() - see
scripts/pipeline.py for the download -> train -> serve orchestrator that
shares this same implementation.

Usage (from the repo root):
    python software/scripts/train.py --config software/config/arlington.yaml
    python software/scripts/train.py --config software/config/seattle.yaml \\
        --model-type vit --epochs 20
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loaf.pipeline import train_stage  # noqa: E402

logger = logging.getLogger(__name__)


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
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating the HTML training report (report.html in the output dir) "
        "that's otherwise built after every run.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

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
    logger.info(f"Best checkpoint: {checkpoint_path}")

    if not args.no_report:
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


if __name__ == "__main__":
    main()
