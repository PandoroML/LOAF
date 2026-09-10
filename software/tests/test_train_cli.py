"""Tests for the loaf-train CLI (scripts/train.py).

Spawns the real script as a subprocess (same pattern as test_pipeline.py) so
it exercises the exact thing a user runs, including the default-on report
generation.
"""

import subprocess
import sys
from pathlib import Path

TRAIN_SCRIPT = Path(__file__).parent.parent / "scripts" / "train.py"

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


def run_train(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


class TestTrainCLI:
    def test_generates_report_by_default(
        self, tmp_path: Path, synthetic_arlington_data: tuple[Path, int]
    ) -> None:
        data_dir, year = synthetic_arlington_data
        config_path = tmp_path / "config.yaml"
        config_path.write_text(CONFIG_YAML)
        output_dir = tmp_path / "runs"

        result = run_train(
            "--config", str(config_path),
            "--data-dir", str(data_dir),
            "--year", str(year),
            "--output-dir", str(output_dir),
            "--device", "cpu",
        )

        assert result.returncode == 0, result.stderr
        assert (output_dir / "best.pt").exists()
        assert (output_dir / "report.html").exists()
        assert "Report written to" in result.stderr

    def test_no_report_skips_report_generation(
        self, tmp_path: Path, synthetic_arlington_data: tuple[Path, int]
    ) -> None:
        data_dir, year = synthetic_arlington_data
        config_path = tmp_path / "config.yaml"
        config_path.write_text(CONFIG_YAML)
        output_dir = tmp_path / "runs"

        result = run_train(
            "--config", str(config_path),
            "--data-dir", str(data_dir),
            "--year", str(year),
            "--output-dir", str(output_dir),
            "--device", "cpu",
            "--no-report",
        )

        assert result.returncode == 0, result.stderr
        assert (output_dir / "best.pt").exists()
        assert not (output_dir / "report.html").exists()
