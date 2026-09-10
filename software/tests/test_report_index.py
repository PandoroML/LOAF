"""Tests for loaf.reporting's cross-run "master report" (index/summary).

Trains two tiny MPNN runs via loaf.pipeline.train_stage (same fixture
pattern as test_reporting.py), then verifies discover_runs()/
build_index_data()/render_index_html() summarize them correctly.
"""

from pathlib import Path

import pytest
from loaf.pipeline import train_stage
from loaf.reporting import build_index_data, discover_runs, render_index_html
from loaf.reporting.collect import build_report_data
from loaf.reporting.render import render_html

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
  epochs: 2
  batch_size: 4
  val_split: 0.2
"""


@pytest.fixture
def two_runs(tmp_path: Path, synthetic_arlington_data: tuple[Path, int]) -> tuple[Path, Path]:
    """Two trained run directories under a shared runs_dir; return (runs_dir, data_dir)."""
    data_dir, year = synthetic_arlington_data
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(CONFIG_YAML)
    runs_dir = tmp_path / "runs"

    for name in ("run-a", "run-b"):
        train_stage(
            config_path=config_path,
            data_dir=data_dir,
            output_dir=runs_dir / name,
            year=year,
            device="cpu",
        )
    return runs_dir, data_dir


class TestDiscoverRuns:
    def test_finds_both_run_dirs(self, two_runs: tuple[Path, Path]) -> None:
        runs_dir, _data_dir = two_runs

        found = discover_runs(runs_dir)

        assert {d.name for d in found} == {"run-a", "run-b"}

    def test_empty_dir_returns_empty_list(self, tmp_path: Path) -> None:
        assert discover_runs(tmp_path / "nonexistent") == []

    def test_ignores_non_run_subdirectories(self, two_runs: tuple[Path, Path]) -> None:
        runs_dir, _data_dir = two_runs
        (runs_dir / "not-a-run").mkdir()

        found = discover_runs(runs_dir)

        assert "not-a-run" not in {d.name for d in found}


class TestBuildIndexData:
    def test_summarizes_every_run(self, two_runs: tuple[Path, Path]) -> None:
        runs_dir, data_dir = two_runs

        index = build_index_data(runs_dir, data_dir=data_dir)

        assert len(index.runs) == 2
        names = {r.run_name for r in index.runs}
        assert names == {"run-a", "run-b"}
        for run in index.runs:
            assert run.error is None
            assert run.model_type == "mpnn"
            assert run.training_curve  # comes from train_log.csv, fast path
            assert run.final_metrics.get("rmse") is not None

    def test_fast_path_works_without_a_data_dir(self, two_runs: tuple[Path, Path]) -> None:
        runs_dir, _data_dir = two_runs

        # Default run_inference_eval=False never touches data_dir, so this
        # succeeds even when the training data isn't available at all -
        # metrics fall back to train_log.csv's best-epoch row.
        index = build_index_data(runs_dir, data_dir=None)

        for run in index.runs:
            assert run.error is None
            assert run.final_metrics.get("rmse") is not None

    def test_one_corrupt_run_does_not_sink_the_index(self, two_runs: tuple[Path, Path]) -> None:
        runs_dir, data_dir = two_runs
        (runs_dir / "run-a" / "best.pt").write_bytes(b"not a checkpoint")
        (runs_dir / "run-a" / "last.pt").write_bytes(b"not a checkpoint")

        index = build_index_data(runs_dir, data_dir=data_dir)

        by_name = {r.run_name: r for r in index.runs}
        assert by_name["run-a"].error is not None
        assert by_name["run-b"].error is None


class TestRenderIndexHtml:
    def test_writes_html_with_run_table_and_links(
        self, two_runs: tuple[Path, Path], tmp_path: Path
    ) -> None:
        runs_dir, data_dir = two_runs
        for name in ("run-a", "run-b"):
            report_data = build_report_data(runs_dir / name, data_dir=data_dir)
            render_html(report_data, runs_dir / name / "report.html")

        index = build_index_data(runs_dir, data_dir=data_dir)
        output_path = render_index_html(index, runs_dir / "summary.html")

        html = output_path.read_text()
        assert "run-a" in html
        assert "run-b" in html
        assert "run-a/report.html" in html
        assert "run-b/report.html" in html
        assert "index-data" in html

    def test_handles_zero_runs_gracefully(self, tmp_path: Path) -> None:
        index = build_index_data(tmp_path / "empty-runs", data_dir=None)

        output_path = render_index_html(index, tmp_path / "summary.html")

        html = output_path.read_text()
        assert "No runs found" in html
