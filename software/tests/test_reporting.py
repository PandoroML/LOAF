"""Tests for loaf.reporting - training report data assembly and HTML rendering.

Trains a tiny MPNN via loaf.pipeline.train_stage on synthetic Arlington, VA
station data (same fixture pattern as test_predictor.py), then verifies
build_report_data() and render_html() produce a sensible report both with
and without re-running validation inference.
"""

from pathlib import Path

import pytest
from loaf.pipeline import train_stage
from loaf.reporting import build_report_data, render_html
from loaf.reporting.collect import _build_tuning_notes

# Small/fast variant of config/arlington.yaml, sized to synthetic_arlington_data.
CONFIG_YAML = """
region:
  name: arlington-test
  lat_min: 38.5
  lat_max: 39.5
  lon_min: -78.0
  lon_max: -76.5
data:
  back_hrs: 6
  lead_times: [3, 6]
model:
  type: mpnn
  hidden_dim: 8
  num_gnn_layers: 1
training:
  epochs: 3
  batch_size: 4
  val_split: 0.2
  patience: 10
"""


@pytest.fixture
def trained_run(tmp_path: Path, synthetic_arlington_data: tuple[Path, int]) -> tuple[Path, Path]:
    """Train a tiny MPNN via train_stage(); return (run_dir, data_dir)."""
    data_dir, year = synthetic_arlington_data
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(CONFIG_YAML)

    checkpoint_path, _state = train_stage(
        config_path=config_path,
        data_dir=data_dir,
        output_dir=tmp_path / "runs" / "arlington-test",
        year=year,
        device="cpu",
    )
    return checkpoint_path.parent, data_dir


class TestBuildReportData:
    def test_includes_training_curve_and_hyperparams(self, trained_run) -> None:
        run_dir, data_dir = trained_run

        data = build_report_data(run_dir, data_dir=data_dir)

        assert data.inference_error is None
        assert len(data.training_curve) == 3
        assert data.training_curve[0]["epoch"] == 1
        assert data.best_epoch in (1, 2, 3)
        assert data.model_type == "mpnn"
        assert data.hyperparams["region"] == "arlington-test"
        assert data.hyperparams["model.hidden_dim"] == 8
        assert set(data.target_vars) == {"u", "v"}

    def test_per_horizon_and_scatter_populated_when_inference_runs(self, trained_run) -> None:
        run_dir, data_dir = trained_run

        data = build_report_data(run_dir, data_dir=data_dir)

        assert set(data.per_horizon) == {"u", "v"}
        for rows in data.per_horizon.values():
            assert [r["lead_hr"] for r in rows] == [3, 6]
            for row in rows:
                assert row["rmse"] >= 0
        for var in data.target_vars:
            assert len(data.scatter[var]) > 0
            assert data.residuals[var]["bin_edges"]

    def test_skips_inference_gracefully_when_data_missing(
        self, trained_run, tmp_path: Path
    ) -> None:
        run_dir, _data_dir = trained_run

        data = build_report_data(run_dir, data_dir=tmp_path / "nonexistent-data")

        assert data.inference_error is not None
        assert data.per_horizon == {}
        assert data.training_curve  # still comes from train_log.csv

    def test_no_inference_flag_skips_inference_without_data_dir(self, trained_run) -> None:
        run_dir, _data_dir = trained_run

        data = build_report_data(run_dir, data_dir=None, run_inference_eval=False)

        assert data.inference_error is None
        assert data.per_horizon == {}
        assert data.training_curve

    def test_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            build_report_data(tmp_path / "empty-run", data_dir=None, run_inference_eval=False)


class TestRenderHtml:
    def test_writes_self_contained_html(self, trained_run, tmp_path: Path) -> None:
        run_dir, data_dir = trained_run
        data = build_report_data(run_dir, data_dir=data_dir)

        output_path = render_html(data, tmp_path / "report.html")

        assert output_path.exists()
        html = output_path.read_text()
        assert "<title>" in html
        assert "mpnn" in html.lower()
        assert data.run_name in html
        # No CDN/network dependencies - the report must work offline. (The SVG
        # XML namespace URI is a literal string, not a fetched resource.)
        assert "<script src=" not in html and "<link " not in html
        assert "cdn." not in html
        assert "report-data" in html  # embedded JSON payload
        assert "</script>" in html

    def test_renders_without_inference_data(self, trained_run, tmp_path: Path) -> None:
        run_dir, _data_dir = trained_run
        data = build_report_data(run_dir, data_dir=None, run_inference_eval=False)

        output_path = render_html(data, tmp_path / "report.html")

        html = output_path.read_text()
        assert "Accuracy by forecast horizon" not in html
        assert "No obvious red flags" in html or "Skill score" in html or "overfitting" in html


class TestTuningNotes:
    def test_flags_negative_skill_as_critical(self) -> None:
        notes = _build_tuning_notes(
            training_curve=[{"epoch": 1, "train_loss": 1.0, "val_loss": 1.0}],
            final_metrics={"skill": -0.5},
            per_horizon={},
        )

        assert any(n["severity"] == "critical" for n in notes)

    def test_flags_overfitting(self) -> None:
        notes = _build_tuning_notes(
            training_curve=[{"epoch": 1, "train_loss": 1.0, "val_loss": 5.0}],
            final_metrics={},
            per_horizon={},
        )

        assert any(n["severity"] == "warning" and "overfitting" in n["text"] for n in notes)

    def test_flags_horizon_degradation(self) -> None:
        notes = _build_tuning_notes(
            training_curve=[],
            final_metrics={},
            per_horizon={"u": [{"lead_hr": 6, "rmse": 1.0}, {"lead_hr": 48, "rmse": 3.0}]},
        )

        assert any("Long-horizon" in n["text"] for n in notes)

    def test_no_flags_reports_good(self) -> None:
        notes = _build_tuning_notes(
            training_curve=[{"epoch": 1, "train_loss": 1.0, "val_loss": 1.05}],
            final_metrics={"skill": 0.5},
            per_horizon={"u": [{"lead_hr": 6, "rmse": 1.0}, {"lead_hr": 12, "rmse": 1.1}]},
        )

        assert any(n["severity"] == "good" for n in notes)
