"""Tests for HRRR forecast mode (grid input over the lead hours).

As in LocalizedWeather, the HRRR grid input is the analysis (step 0) for each
historical hour plus the forecast steps 1..max(lead_times) from the run
initialized at "now". Uses a synthetic HRRR file whose values encode
(time index, step) so the selected hours can be checked exactly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loaf.data.loaders.dataset import create_dataloaders
from loaf.data.loaders.hrrr import HRRRLoader
from loaf.inference import Predictor
from loaf.pipeline import train_stage

CONFIG_YAML = """
region:
  name: arlington-test
  lat_min: 38.5
  lat_max: 39.5
  lon_min: -78.0
  lon_max: -76.5
data:
  back_hrs: 6
  lead_times: [1, 3]
model:
  type: mpnn
  hidden_dim: 8
  num_gnn_layers: 1
training:
  epochs: 1
  batch_size: 4
  val_split: 0.2
"""


def make_synthetic_hrrr_data(
    hrrr_dir: Path, year: int, n_hours: int = 200, n_steps: int = 4
) -> None:
    """Write one HRRR-format NetCDF file (time, step, y, x) with one NaN cell.

    Every variable's value is time_index * 100 + step, so a sample's values
    identify exactly which (init time, step) they came from.
    """
    hrrr_dir.mkdir(parents=True, exist_ok=True)
    times = pd.date_range(f"{year}-01-01", periods=n_hours, freq="h")
    lats, lons = np.meshgrid(
        np.linspace(38.5, 39.5, 4), np.linspace(282.0, 283.5, 5), indexing="ij"
    )

    values = (
        np.arange(n_hours)[:, None, None, None] * 100.0
        + np.arange(n_steps)[None, :, None, None]
        + np.zeros((1, 1, *lats.shape))
    ).astype(np.float32)
    values[:, :, 0, 0] = np.nan  # outside the true lat/lon box, as in real downloads

    dims = ("time", "step", "y", "x")
    xr.Dataset(
        {name: (dims, values) for name in ["u10", "v10", "t2m", "d2m"]},
        coords={
            "time": times,
            "step": pd.to_timedelta(np.arange(n_steps), unit="h"),
            "latitude": (("y", "x"), lats),
            "longitude": (("y", "x"), lons),
        },
    ).to_netcdf(hrrr_dir / f"hrrr_{year}0101.nc")


@pytest.fixture
def synthetic_hrrr_data(synthetic_arlington_data: tuple[Path, int]) -> tuple[Path, int]:
    data_dir, year = synthetic_arlington_data
    make_synthetic_hrrr_data(data_dir / "hrrr", year)
    return data_dir, year


class TestGetSampleWithForecast:
    def _loader(self, data_dir: Path, year: int) -> HRRRLoader:
        loader = HRRRLoader(
            data_dir / "hrrr",
            years=[year],
            lat_bounds=(38.5, 39.5),
            lon_bounds=(-78.0, -76.5),
            reanalysis_only=False,
        )
        loader.load_to_memory()
        return loader

    def test_history_is_analysis_and_future_is_run_at_now(self, synthetic_hrrr_data) -> None:
        data_dir, year = synthetic_hrrr_data
        loader = self._loader(data_dir, year)
        times = loader.times

        # History: times 10..15 ("now" = 15); forecast: steps 1..3 of the 15Z run.
        sample = loader.get_sample_with_forecast(times[10], times[15], 6, 3, ["u"])

        assert sample["u"].shape == (loader.n_nodes, 9)
        expected = [t * 100.0 for t in range(10, 16)] + [1501.0, 1502.0, 1503.0]
        np.testing.assert_array_equal(sample["u"][0].numpy(), expected)

    def test_excludes_nan_cells(self, synthetic_hrrr_data) -> None:
        data_dir, year = synthetic_hrrr_data
        loader = self._loader(data_dir, year)
        times = loader.times

        sample = loader.get_sample_with_forecast(times[10], times[15], 6, 3, ["u"])

        assert loader.n_nodes == 19
        assert not sample["u"].isnan().any()

    def test_raises_when_steps_missing(self, synthetic_hrrr_data) -> None:
        data_dir, year = synthetic_hrrr_data
        loader = self._loader(data_dir, year)
        times = loader.times

        with pytest.raises(ValueError, match="max_lead_hr >= 4"):
            loader.get_sample_with_forecast(times[10], times[15], 6, 4, ["u"])


class TestDatasetForecastMode:
    def test_grid_input_spans_history_and_lead_hours(self, synthetic_hrrr_data) -> None:
        data_dir, year = synthetic_hrrr_data
        bundle = create_dataloaders(
            data_dir,
            year,
            back_hrs=6,
            lead_times=[1, 3],
            lat_bounds=(38.5, 39.5),
            lon_bounds=(-78.0, -76.5),
            use_hrrr=True,
        )
        dataset = bundle.dataset

        assert dataset.grid_len == 9
        sample = dataset[20]
        assert sample["ex_x"].shape == (19, 9, 4)
        assert not sample["ex_x"].isnan().any()

    def test_raises_when_hrrr_steps_do_not_cover_lead_times(
        self, synthetic_arlington_data
    ) -> None:
        data_dir, year = synthetic_arlington_data
        make_synthetic_hrrr_data(data_dir / "hrrr", year, n_steps=2)

        with pytest.raises(ValueError, match="--max-lead-hr 3"):
            create_dataloaders(
                data_dir,
                year,
                back_hrs=6,
                lead_times=[1, 3],
                lat_bounds=(38.5, 39.5),
                lon_bounds=(-78.0, -76.5),
                use_hrrr=True,
            )


def test_train_and_predict_with_hrrr_forecasts(tmp_path: Path, synthetic_hrrr_data) -> None:
    data_dir, year = synthetic_hrrr_data
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(CONFIG_YAML)

    checkpoint_path, _state = train_stage(
        config_path=config_path,
        data_dir=data_dir,
        output_dir=tmp_path / "runs",
        year=year,
        use_hrrr=True,
        device="cpu",
    )
    predictor = Predictor(checkpoint_path, data_dir=data_dir, year=year, device="cpu")

    assert predictor.in_hrs_grid == 9
    for forecast in predictor.predict_all(force_refresh=True):
        for lead in (1, 3):
            assert all(np.isfinite(v) for v in forecast.values[lead].values())
