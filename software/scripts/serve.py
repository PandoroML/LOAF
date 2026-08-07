#!/usr/bin/env python3
"""Forecast API server entry point for LOAF weather forecasting models.

Loads a trained checkpoint (see scripts/train.py) and serves multi-horizon
wind forecasts over REST, keyed by (lat, lon).

Usage (from the repo root):
    python software/scripts/serve.py --checkpoint runs/arlington_.../best.pt
    python software/scripts/serve.py --checkpoint runs/arlington_.../best.pt \\
        --data-dir data --port 5000

Verify it's working:
    curl http://localhost:5000/health
    curl "http://localhost:5000/api/forecast?lat=38.88&lon=-77.10"   # Arlington, VA
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loaf.inference.predictor import Predictor  # noqa: E402
from loaf.inference.server import run_server  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve LOAF wind forecasts over REST from a trained checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        required=True,
        help="Path to a checkpoint written by Trainer (e.g. runs/<run>/best.pt).",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Base directory containing downloaded data (default: data)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year of station data to serve (default: latest year with downloaded IEM data)",
    )
    parser.add_argument(
        "--cache-ttl",
        type=float,
        default=300.0,
        help="Seconds to reuse a forward pass before recomputing it (default: 300)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Interface to bind (default: 0.0.0.0)"
    )
    parser.add_argument("--port", type=int, default=5000, help="Port to bind (default: 5000)")
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run inference on (default: cuda if available, else cpu)",
    )
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug/reload mode")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Loading checkpoint {args.checkpoint}")
    predictor = Predictor(
        checkpoint_path=args.checkpoint,
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
