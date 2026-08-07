"""Tests for loaf.pipeline and the loaf-pipeline CLI (scripts/pipeline.py).

The CLI tests spawn the real script as a subprocess (rather than importing
main() and mocking sys.argv) so they exercise the exact thing a user runs.
"""

import subprocess
import sys
from pathlib import Path

import pytest
from loaf.pipeline import _to_hrrr_lon

PIPELINE_SCRIPT = Path(__file__).parent.parent / "scripts" / "pipeline.py"

# Small/fast config, sized to the synthetic_arlington_data fixture (see conftest.py).
CONFIG_YAML = """
region:
  name: arlington-test
  lat_min: 38.5
  lat_max: 39.5
  lon_min: -78.0
  lon_max: -76.5
data:
  back_hrs: 6
  lead_times: [3]
model:
  type: mpnn
  hidden_dim: 8
  num_gnn_layers: 1
training:
  epochs: 1
  batch_size: 4
  val_split: 0.2
"""


def run_pipeline(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PIPELINE_SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


class TestHelpers:
    def test_to_hrrr_lon_converts_negative_longitudes(self) -> None:
        assert _to_hrrr_lon(-77.0) == 283.0
        assert _to_hrrr_lon(-124.0) == 236.0

    def test_to_hrrr_lon_leaves_positive_longitudes(self) -> None:
        assert _to_hrrr_lon(283.0) == 283.0


class TestCLIValidation:
    """Argument validation should fail fast, before touching the network or disk."""

    def test_requires_dates_unless_skip_download(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(CONFIG_YAML)

        result = run_pipeline("--config", str(config_path))

        assert result.returncode == 2
        assert "--start-date and --end-date are required" in result.stderr

    def test_requires_checkpoint_when_skip_train(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(CONFIG_YAML)

        result = run_pipeline(
            "--config", str(config_path), "--skip-download", "--skip-train"
        )

        assert result.returncode == 2
        assert "--checkpoint is required" in result.stderr

    def test_help_exits_cleanly(self) -> None:
        result = run_pipeline("--help")

        assert result.returncode == 0
        assert "download data, train a model, and serve forecasts" in result.stdout.lower()


class TestPipelineRun:
    """--skip-download + --no-serve: exercises train_stage through the real CLI."""

    def test_skip_download_trains_and_writes_checkpoint(
        self, tmp_path: Path, synthetic_arlington_data: tuple[Path, int]
    ) -> None:
        data_dir, year = synthetic_arlington_data
        config_path = tmp_path / "config.yaml"
        config_path.write_text(CONFIG_YAML)
        output_dir = tmp_path / "runs"

        result = run_pipeline(
            "--config", str(config_path),
            "--data-dir", str(data_dir),
            "--skip-download",
            "--no-serve",
            "--year", str(year),
            "--output-dir", str(output_dir),
            "--device", "cpu",
        )

        assert result.returncode == 0, result.stderr
        assert (output_dir / "best.pt").exists()
        assert "[1/3] Skipping download" in result.stderr
        assert "[3/3] Skipping serve" in result.stderr

    def test_skip_download_and_skip_train_serves_existing_checkpoint(
        self, tmp_path: Path, synthetic_arlington_data: tuple[Path, int]
    ) -> None:
        # First, produce a real checkpoint the same way the previous test does.
        data_dir, year = synthetic_arlington_data
        config_path = tmp_path / "config.yaml"
        config_path.write_text(CONFIG_YAML)
        output_dir = tmp_path / "runs"

        train_result = run_pipeline(
            "--config", str(config_path),
            "--data-dir", str(data_dir),
            "--skip-download",
            "--no-serve",
            "--year", str(year),
            "--output-dir", str(output_dir),
            "--device", "cpu",
        )
        assert train_result.returncode == 0, train_result.stderr
        checkpoint_path = output_dir / "best.pt"
        assert checkpoint_path.exists()

        # Then re-invoke with both --skip-download and --skip-train: it should
        # go straight to loading the checkpoint (we bail with --no-serve
        # right before actually binding a port, since the server blocks).
        serve_result = run_pipeline(
            "--config", str(config_path),
            "--data-dir", str(data_dir),
            "--skip-download",
            "--skip-train",
            "--checkpoint", str(checkpoint_path),
            "--year", str(year),
            "--no-serve",
            "--device", "cpu",
        )

        assert serve_result.returncode == 0, serve_result.stderr
        assert f"Checkpoint: {checkpoint_path}" in serve_result.stderr


@pytest.mark.slow
@pytest.mark.network
class TestDownloadStageNetwork:
    """Hits the real IEM endpoint - excluded from the default `-m "not slow"` run."""

    def test_download_stage_writes_real_iem_data(self, tmp_path: Path) -> None:
        from datetime import datetime

        from loaf.pipeline import download_stage

        config_path = tmp_path / "config.yaml"
        config_path.write_text(CONFIG_YAML)
        data_dir = tmp_path / "data"

        download_stage(
            config_path=config_path,
            data_dir=data_dir,
            start_date=datetime(2024, 10, 1),
            end_date=datetime(2024, 10, 1),
            stations=["DCA"],
        )

        files = list((data_dir / "iem").glob("iem_*.parquet"))
        assert files, "expected at least one downloaded IEM file"
