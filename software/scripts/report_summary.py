#!/usr/bin/env python3
"""Generate a "master report" summarizing every LOAF run so far.

Scans a runs/ directory for every training run (anything with a best.pt or
last.pt), and builds a single standalone HTML page comparing them: a
training-loss overlay, best-RMSE and best-skill-vs-persistence bar charts
across runs, and a sortable table linking out to each run's own report.html
(see scripts/report.py) when one exists.

Fast by default - it reads train_log.csv and checkpoint metadata only, the
same way a single-run report.html falls back when it skips re-running
validation inference. Pass --recompute-metrics for per-run metrics freshly
computed against data_dir instead (slower: one validation pass per run).

Usage (from the repo root):
    python software/scripts/report_summary.py --runs-dir runs
    python software/scripts/report_summary.py --runs-dir runs \\
        --output runs/summary.html --open
"""

import argparse
import logging
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loaf.reporting import build_index_data, render_index_html  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a master HTML report summarizing every LOAF run so far."
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Directory containing run subdirectories (default: runs)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Base directory containing downloaded data, used only with "
        "--recompute-metrics (default: data)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output HTML path (default: <runs-dir>/summary.html).",
    )
    parser.add_argument(
        "--recompute-metrics",
        action="store_true",
        help="Re-run validation inference for every run instead of reading train_log.csv's "
        "best-epoch metrics - more accurate, much slower with many runs.",
    )
    parser.add_argument("--open", action="store_true", help="Open the report when done.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    runs_dir = Path(args.runs_dir)
    output_path = Path(args.output) if args.output else runs_dir / "summary.html"

    data = build_index_data(
        runs_dir=runs_dir,
        data_dir=args.data_dir if args.recompute_metrics else None,
        run_inference_eval=args.recompute_metrics,
    )
    logger.info(f"Found {len(data.runs)} run(s) under {runs_dir}")
    for run in data.runs:
        if run.error:
            logger.warning(f"{run.run_name}: couldn't read this run ({run.error})")

    output_path = render_index_html(data, output_path)
    logger.info(f"Summary written to {output_path}")

    if args.open:
        webbrowser.open(output_path.resolve().as_uri())


if __name__ == "__main__":
    main()
