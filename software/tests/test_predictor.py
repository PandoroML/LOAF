"""End-to-end test for loaf.inference.Predictor and the REST API server.

Trains a tiny MPNN via loaf.pipeline.train_stage on synthetic Arlington, VA
station data (the same function loaf-train and loaf-pipeline call), then
verifies Predictor (and the Flask app built on top of it) return forecasts
for real Arlington coordinates - Milestone 4's verification step.
"""

from pathlib import Path

import numpy as np
import pytest
from loaf.inference import Predictor
from loaf.inference.server import create_app
from loaf.pipeline import train_stage

from tests.conftest import ARLINGTON_STATIONS, REAGAN_NATIONAL

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


@pytest.fixture
def trained_checkpoint(
    tmp_path: Path, synthetic_arlington_data: tuple[Path, int]
) -> tuple[Path, Path, int]:
    """Train a tiny MPNN via train_stage(); return (checkpoint_path, data_dir, year)."""
    data_dir, year = synthetic_arlington_data
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(CONFIG_YAML)

    checkpoint_path, _state = train_stage(
        config_path=config_path,
        data_dir=data_dir,
        output_dir=tmp_path / "runs",
        year=year,
        device="cpu",
    )
    return checkpoint_path, data_dir, year


class TestPredictor:
    """loaf.inference.Predictor tests."""

    def test_predict_returns_forecast_for_arlington_coordinates(
        self, trained_checkpoint
    ) -> None:
        checkpoint_path, data_dir, year = trained_checkpoint
        predictor = Predictor(checkpoint_path, data_dir=data_dir, year=year, device="cpu")

        forecast = predictor.predict(*REAGAN_NATIONAL)

        assert forecast.station_id in ARLINGTON_STATIONS
        assert forecast.lead_times == [3]
        assert set(forecast.target_vars) == {"u", "v"}
        assert set(forecast.values[3]) == {"u", "v"}
        for value in forecast.values[3].values():
            assert np.isfinite(value)

    def test_predict_all_covers_every_station(self, trained_checkpoint) -> None:
        checkpoint_path, data_dir, year = trained_checkpoint
        predictor = Predictor(checkpoint_path, data_dir=data_dir, year=year, device="cpu")

        forecasts = predictor.predict_all()

        assert len(forecasts) == predictor.n_stations
        assert {f.station_id for f in forecasts} <= set(ARLINGTON_STATIONS)

    def test_predict_caches_between_calls(self, trained_checkpoint) -> None:
        checkpoint_path, data_dir, year = trained_checkpoint
        predictor = Predictor(
            checkpoint_path, data_dir=data_dir, year=year, device="cpu", cache_ttl=60.0
        )

        first = predictor.predict_all()
        second = predictor.predict_all()

        assert first is second  # same cached list object, no recomputation

        third = predictor.predict_all(force_refresh=True)
        assert third is not first

    def test_predict_out_of_region_raises(self, trained_checkpoint) -> None:
        checkpoint_path, data_dir, year = trained_checkpoint
        predictor = Predictor(checkpoint_path, data_dir=data_dir, year=year, device="cpu")

        with pytest.raises(ValueError):
            predictor.predict(0.0, 0.0)  # nowhere near Arlington, VA


class TestServer:
    """loaf.inference.server (Flask app) tests."""

    def test_api_forecast_endpoint(self, trained_checkpoint) -> None:
        checkpoint_path, data_dir, year = trained_checkpoint
        predictor = Predictor(checkpoint_path, data_dir=data_dir, year=year, device="cpu")
        client = create_app(predictor).test_client()

        response = client.get(
            "/api/forecast", query_string={"lat": REAGAN_NATIONAL[0], "lon": REAGAN_NATIONAL[1]}
        )

        assert response.status_code == 200
        body = response.get_json()
        assert body["station_id"] in ARLINGTON_STATIONS
        assert body["forecasts"][0]["lead_hr"] == 3
        assert "wind_speed" in body["forecasts"][0]
        assert "wind_direction" in body["forecasts"][0]

    def test_api_forecast_missing_params(self, trained_checkpoint) -> None:
        checkpoint_path, data_dir, year = trained_checkpoint
        predictor = Predictor(checkpoint_path, data_dir=data_dir, year=year, device="cpu")
        client = create_app(predictor).test_client()

        assert client.get("/api/forecast").status_code == 400

    def test_api_forecast_bad_lat_lon(self, trained_checkpoint) -> None:
        checkpoint_path, data_dir, year = trained_checkpoint
        predictor = Predictor(checkpoint_path, data_dir=data_dir, year=year, device="cpu")
        client = create_app(predictor).test_client()

        response = client.get("/api/forecast", query_string={"lat": "nope", "lon": "-77.0"})
        assert response.status_code == 400

    def test_api_forecast_all(self, trained_checkpoint) -> None:
        checkpoint_path, data_dir, year = trained_checkpoint
        predictor = Predictor(checkpoint_path, data_dir=data_dir, year=year, device="cpu")
        client = create_app(predictor).test_client()

        response = client.get("/api/forecast/all")

        assert response.status_code == 200
        assert len(response.get_json()["stations"]) == predictor.n_stations

    def test_api_stations(self, trained_checkpoint) -> None:
        checkpoint_path, data_dir, year = trained_checkpoint
        predictor = Predictor(checkpoint_path, data_dir=data_dir, year=year, device="cpu")
        client = create_app(predictor).test_client()

        response = client.get("/api/stations")

        assert response.status_code == 200
        ids = {s["station_id"] for s in response.get_json()["stations"]}
        assert ids <= set(ARLINGTON_STATIONS)

    def test_health_endpoint(self, trained_checkpoint) -> None:
        checkpoint_path, data_dir, year = trained_checkpoint
        predictor = Predictor(checkpoint_path, data_dir=data_dir, year=year, device="cpu")
        client = create_app(predictor).test_client()

        response = client.get("/health")

        assert response.status_code == 200
        assert response.get_json()["model_type"] == "mpnn"
