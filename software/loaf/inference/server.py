"""REST API server for LOAF wind forecasts.

Wraps a loaf.inference.Predictor in a small Flask app, so a trained
checkpoint can be queried for a forecast at any (lat, lon) - e.g. for a
Home Assistant REST sensor (see plan/dev-plan-ml-pipeline.md Phase 5).

Run via scripts/serve.py rather than importing this module directly.
"""

from __future__ import annotations

import logging
from typing import Any

from flask import Flask, jsonify, request

from loaf.inference.predictor import Predictor, StationForecast

logger = logging.getLogger(__name__)


def _with_wind_speed_direction(forecast: StationForecast) -> dict[str, Any]:
    """StationForecast.to_dict(), plus derived wind_speed/wind_direction.

    Only added when the model predicts u/v wind components - the format
    Home Assistant (and most weather UIs) expect, versus raw components.
    """
    payload = forecast.to_dict()
    if "u" not in forecast.target_vars or "v" not in forecast.target_vars:
        return payload

    import math

    for entry in payload["forecasts"]:
        u, v = entry["u"], entry["v"]
        entry["wind_speed"] = math.hypot(u, v)
        # Meteorological convention: direction wind is blowing FROM, clockwise from north.
        entry["wind_direction"] = (math.degrees(math.atan2(-u, -v))) % 360
    return payload


def create_app(predictor: Predictor) -> Flask:
    """Build the Flask app for a given (already-loaded) Predictor.

    Args:
        predictor: A loaf.inference.Predictor with a checkpoint and data loaded.

    Returns:
        A Flask app exposing:
        - GET  /health              - liveness/model info
        - GET  /api/forecast        - nearest-station forecast for ?lat=&lon=
        - GET  /api/forecast/all    - forecast for every station in the graph
        - GET  /api/stations        - station id/lat/lon listing
        - POST /api/refresh         - force-reload data from disk
    """
    app = Flask(__name__)

    @app.get("/health")
    def health() -> Any:
        return jsonify(
            {
                "status": "ok",
                "model_type": predictor.model_type,
                "checkpoint": str(predictor.checkpoint_path),
                "n_stations": predictor.n_stations,
                "lead_times": predictor.lead_times,
                "target_vars": predictor.target_vars,
                "region": {"lat_bounds": predictor.lat_bounds, "lon_bounds": predictor.lon_bounds},
            }
        )

    @app.get("/api/forecast")
    def forecast() -> Any:
        lat_raw, lon_raw = request.args.get("lat"), request.args.get("lon")
        if lat_raw is None or lon_raw is None:
            return jsonify({"error": "Query params 'lat' and 'lon' are required."}), 400

        try:
            lat, lon = float(lat_raw), float(lon_raw)
        except ValueError:
            return jsonify({"error": "'lat' and 'lon' must be numbers."}), 400

        force_refresh = request.args.get("refresh", "").lower() in ("1", "true", "yes")

        try:
            result = predictor.predict(lat, lon, force_refresh=force_refresh)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except RuntimeError as e:
            logger.exception("Prediction failed")
            return jsonify({"error": str(e)}), 503

        return jsonify(_with_wind_speed_direction(result))

    @app.get("/api/forecast/all")
    def forecast_all() -> Any:
        force_refresh = request.args.get("refresh", "").lower() in ("1", "true", "yes")
        try:
            results = predictor.predict_all(force_refresh=force_refresh)
        except RuntimeError as e:
            logger.exception("Prediction failed")
            return jsonify({"error": str(e)}), 503

        return jsonify({"stations": [_with_wind_speed_direction(r) for r in results]})

    @app.get("/api/stations")
    def stations() -> Any:
        meta = predictor.station_metadata
        return jsonify(
            {
                "stations": [
                    {"station_id": sid, "lat": lat, "lon": lon}
                    for sid, lat, lon in zip(
                        meta.station_ids, meta.lats.tolist(), meta.lons.tolist()
                    )
                ]
            }
        )

    @app.post("/api/refresh")
    def refresh() -> Any:
        try:
            predictor.refresh_data()
        except RuntimeError as e:
            logger.exception("Refresh failed")
            return jsonify({"error": str(e)}), 503
        return jsonify({"status": "refreshed", "year": predictor.year})

    @app.errorhandler(404)
    def not_found(_e: Any) -> Any:
        return jsonify({"error": "Not found"}), 404

    return app


def run_server(
    predictor: Predictor,
    host: str = "0.0.0.0",
    port: int = 5000,
    debug: bool = False,
) -> None:
    """Build and run the forecast API server (blocking)."""
    app = create_app(predictor)
    logger.info(f"Serving LOAF forecasts on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
