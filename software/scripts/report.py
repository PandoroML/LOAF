#!/usr/bin/env python3
"""Generate a standalone HTML training report for a LOAF run.

Reads a run directory written by `loaf-train`/`loaf-pipeline` (train_log.csv
plus best.pt/last.pt) and, when the training data is still on disk, re-runs
the checkpoint's own validation split to add a per-horizon accuracy
breakdown, predicted-vs-actual scatter plots, and residual distributions -
so a developer can see how accurate the model is and where to focus
fine-tuning (see the "Fine-tuning notes" section of the report).

Usage (from the repo root):
    python software/scripts/report.py --run runs/arlington_20260806_201234
    python software/scripts/report.py --run runs/arlington_20260806_201234 \\
        --data-dir data --output runs/arlington_20260806_201234/report.html
    python software/scripts/report.py --run runs/arlington_20260806_201234 \\
        --no-inference  # fast, metrics-only report from train_log.csv alone
"""

import argparse
import logging
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loaf.reporting import build_report_data, render_html  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a standalone HTML training report for a LOAF run."
    )
    parser.add_argument(
        "--run",
        "-r",
        required=True,
        help="Run directory written by loaf-train/loaf-pipeline (contains best.pt/last.pt).",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Base directory containing downloaded data, for re-running validation "
        "inference (default: data). Ignored with --no-inference.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output HTML path (default: <run>/report.html).",
    )
    parser.add_argument(
        "--no-inference",
        action="store_true",
        help="Skip re-running validation inference - builds a fast, data-independent "
        "report from train_log.csv alone (no per-horizon/scatter/residual sections).",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated report in the default browser when done.",
    )
    parser.add_argument(
        "--device", default=None, help="Device for inference (default: cuda if available, else cpu)"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    run_dir = Path(args.run)
    output_path = Path(args.output) if args.output else run_dir / "report.html"

    data = build_report_data(
        run_dir=run_dir,
        data_dir=None if args.no_inference else args.data_dir,
        run_inference_eval=not args.no_inference,
        device=args.device,
    )
    if data.inference_error:
        logger.warning(
            f"Validation inference failed, report will skip that detail: {data.inference_error}"
        )

    output_path = render_html(data, output_path)
    logger.info(f"Report written to {output_path}")

    if args.open:
        webbrowser.open(output_path.resolve().as_uri())


if __name__ == "__main__":
    main()
