"""Tests for HRRRLoader's grid-validity masking and NaN guarding.

Regression coverage for a bug found re-running the HRRR download live: one
(init_time, forecast_step) slice came back entirely NaN for every variable
(see test_hrrr_download.py's TestDownloadHRRRHourlyRobustness for the
download-side fix). Before this fix, _get_valid_mask required a grid cell
to be non-NaN in *every* loaded slice, so that one bad slice zeroed out
validity for every cell in the whole region - which then crashed deep in
unrelated code (WeatherDataset._compute_grid_to_station_edges) with no hint
about the real cause. These tests build small synthetic HRRR-shaped
datasets directly (bypassing the download layer) to exercise that logic
without going over the network.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from loaf.data.loaders.hrrr import HRRRLoader


def _write_hrrr_file(
    path: Path,
    n_times: int = 24,
    n_steps: int = 2,
    ny: int = 4,
    nx: int = 4,
    nan_slices: list[tuple[int, int]] | None = None,
    lat_range: tuple[float, float] = (38.0, 39.5),
    lon_range: tuple[float, float] = (236.0, 238.0),
) -> None:
    """Write a daily HRRR-shaped file with every cell valid, except any
    (time_idx, step_idx) pairs listed in nan_slices, which come back
    all-NaN for every variable - simulating the live gap this module's
    fix is meant to tolerate (at the masking level) or reject (if it falls
    inside a requested sample window)."""
    lats = np.linspace(*lat_range, ny)
    lons = np.linspace(*lon_range, nx)
    lon2d, lat2d = np.meshgrid(lons, lats)

    start = datetime(2024, 10, 1)
    times = [start + timedelta(hours=h) for h in range(n_times)]
    steps = np.arange(n_steps)

    rng = np.random.default_rng(0)
    data_vars = {}
    for raw in ["u10", "v10", "t2m", "d2m"]:
        arr = rng.normal(0, 1, size=(n_times, n_steps, ny, nx)).astype("float32")
        for t_idx, s_idx in nan_slices or []:
            arr[t_idx, s_idx] = np.nan
        data_vars[raw] = (["time", "step", "y", "x"], arr)

    ds = xr.Dataset(
        data_vars,
        coords=dict(
            time=times,
            step=steps,
            latitude=(["y", "x"], lat2d.astype("float32")),
            longitude=(["y", "x"], lon2d.astype("float32")),
        ),
    )
    ds.to_netcdf(path)


class TestValidMaskTolerance:
    """_get_valid_mask should tolerate a small fraction of bad slices
    rather than zeroing out the whole region over one of them."""

    def test_one_bad_slice_does_not_zero_out_the_region(self, tmp_path: Path) -> None:
        _write_hrrr_file(tmp_path / "hrrr_20241001.nc", n_times=24, n_steps=2, nan_slices=[(3, 1)])

        loader = HRRRLoader(tmp_path, years=[2024], lat_bounds=(38.0, 39.5), lon_bounds=(-78, -76))
        loader.load_to_memory()

        # 1 bad slice out of 24*2=48 is well within the tolerance - every
        # structurally-valid cell should still count as valid.
        assert loader.n_nodes > 0

    def test_mask_still_rejects_structurally_out_of_bounds_cells(self, tmp_path: Path) -> None:
        """Cells permanently outside the lat/lon box (NaN in every single
        slice) are a real, static boundary - not a transient gap - and
        should still be excluded."""
        rng = np.random.default_rng(0)
        n_times, n_steps, ny, nx = 10, 2, 4, 4
        lats = np.linspace(38.0, 39.5, ny)
        lons = np.linspace(236.0, 238.0, nx)
        lon2d, lat2d = np.meshgrid(lons, lats)
        start = datetime(2024, 10, 1)

        data_vars = {}
        for raw in ["u10", "v10", "t2m", "d2m"]:
            arr = rng.normal(0, 1, size=(n_times, n_steps, ny, nx)).astype("float32")
            arr[:, :, 0, 0] = np.nan  # one corner cell permanently outside the true footprint
            data_vars[raw] = (["time", "step", "y", "x"], arr)

        ds = xr.Dataset(
            data_vars,
            coords=dict(
                time=[start + timedelta(hours=h) for h in range(n_times)],
                step=np.arange(n_steps),
                latitude=(["y", "x"], lat2d.astype("float32")),
                longitude=(["y", "x"], lon2d.astype("float32")),
            ),
        )
        ds.to_netcdf(tmp_path / "hrrr_20241001.nc")

        loader = HRRRLoader(tmp_path, years=[2024], lat_bounds=(38.0, 39.5), lon_bounds=(-78, -76))
        loader.load_to_memory()

        assert loader.n_nodes == ny * nx - 1

    def test_mostly_bad_cell_is_excluded(self, tmp_path: Path) -> None:
        """A cell that's NaN in (nearly) every slice is the structural
        case _get_valid_mask exists for, not a one-off gap - it should
        stay excluded even under the fraction threshold."""
        rng = np.random.default_rng(0)
        n_times, n_steps, ny, nx = 20, 2, 3, 3
        lats = np.linspace(38.0, 39.5, ny)
        lons = np.linspace(236.0, 238.0, nx)
        lon2d, lat2d = np.meshgrid(lons, lats)
        start = datetime(2024, 10, 1)

        data_vars = {}
        for raw in ["u10", "v10", "t2m", "d2m"]:
            arr = rng.normal(0, 1, size=(n_times, n_steps, ny, nx)).astype("float32")
            arr[:-1, :, 1, 1] = np.nan  # valid in only 1 of 20 times at this cell
            data_vars[raw] = (["time", "step", "y", "x"], arr)

        ds = xr.Dataset(
            data_vars,
            coords=dict(
                time=[start + timedelta(hours=h) for h in range(n_times)],
                step=np.arange(n_steps),
                latitude=(["y", "x"], lat2d.astype("float32")),
                longitude=(["y", "x"], lon2d.astype("float32")),
            ),
        )
        ds.to_netcdf(tmp_path / "hrrr_20241001.nc")

        loader = HRRRLoader(tmp_path, years=[2024], lat_bounds=(38.0, 39.5), lon_bounds=(-78, -76))
        loader.load_to_memory()

        assert loader.n_nodes == ny * nx - 1


class TestSampleNanGuard:
    """A gap that falls inside a specific requested window must still
    surface loudly, even though the cell it's on counts as 'valid'
    overall under the fraction threshold."""

    def test_get_sample_raises_when_window_hits_the_gap(self, tmp_path: Path) -> None:
        _write_hrrr_file(tmp_path / "hrrr_20241001.nc", n_times=24, n_steps=2, nan_slices=[(3, 0)])
        loader = HRRRLoader(tmp_path, years=[2024], lat_bounds=(38.0, 39.5), lon_bounds=(-78, -76))
        loader.load_to_memory()

        times = loader.times
        with pytest.raises(ValueError, match="missing values"):
            loader.get_sample(times[2], times[4], variables=["u"], step=0)

    def test_get_sample_succeeds_outside_the_gap(self, tmp_path: Path) -> None:
        _write_hrrr_file(tmp_path / "hrrr_20241001.nc", n_times=24, n_steps=2, nan_slices=[(3, 0)])
        loader = HRRRLoader(tmp_path, years=[2024], lat_bounds=(38.0, 39.5), lon_bounds=(-78, -76))
        loader.load_to_memory()

        times = loader.times
        sample = loader.get_sample(times[10], times[12], variables=["u"], step=0)
        assert not np.isnan(sample["u"].numpy()).any()

    def test_get_sample_with_forecast_raises_when_forecast_window_hits_the_gap(
        self, tmp_path: Path
    ) -> None:
        _write_hrrr_file(
            tmp_path / "hrrr_20241001.nc", n_times=24, n_steps=4, nan_slices=[(5, 2)]
        )
        loader = HRRRLoader(tmp_path, years=[2024], lat_bounds=(38.0, 39.5), lon_bounds=(-78, -76))
        loader.load_to_memory()

        times = loader.times
        # Historical window ending at the init time whose step=2 forecast is NaN.
        with pytest.raises(ValueError, match="missing values"):
            loader.get_sample_with_forecast(
                times[0], times[5], n_historical=6, n_forecast=2, variables=["u"]
            )
