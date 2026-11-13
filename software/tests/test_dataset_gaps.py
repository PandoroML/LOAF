"""WeatherDataset must skip samples whose grid window crosses a missing hour.

Regression coverage for a crash hit on the first 3-month HRRR training run:
download_hrrr_hourly drops a whole run_time when any forecast step can't be
fetched completely, which leaves holes in HRRRLoader's time axis (6 of 2,208
init hours were missing). usable_index_range() only bounds the overall
range, so a DataLoader eventually asked for a 24-hour window spanning one of
those holes and HRRRLoader raised "Not enough times in window: 23 < 24"
mid-epoch. usable_indices() excludes exactly those windows.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from loaf.data.loaders import create_dataloaders

BACK_HRS = 6
LEAD_TIMES = [3]
LAT_BOUNDS = (38.0, 39.5)
LON_BOUNDS = (-78.0, -76.0)


def _write_hrrr_with_gap(
    path: Path, start: datetime, n_hours: int, missing_hours: list[int]
) -> None:
    """One HRRR-shaped file covering start..start+n_hours, minus missing_hours."""
    times = [start + timedelta(hours=h) for h in range(n_hours) if h not in missing_hours]
    ny, nx, n_steps = 3, 3, 4
    lon2d, lat2d = np.meshgrid(np.linspace(282.0, 284.0, nx), np.linspace(*LAT_BOUNDS, ny))
    rng = np.random.default_rng(0)
    shape = (len(times), n_steps, ny, nx)
    data_vars = {
        raw: (["time", "step", "y", "x"], rng.normal(size=shape).astype("float32"))
        for raw in ["u10", "v10", "t2m", "d2m"]
    }
    xr.Dataset(
        data_vars,
        coords=dict(
            time=times,
            step=np.arange(n_steps),
            latitude=(["y", "x"], lat2d.astype("float32")),
            longitude=(["y", "x"], lon2d.astype("float32")),
        ),
    ).to_netcdf(path)


def _bundle(synthetic_arlington_data: tuple[Path, int], missing_hours: list[int]):
    data_dir, year = synthetic_arlington_data  # 200 synthetic hourly obs (see conftest)
    n_hours = 200
    (data_dir / "hrrr").mkdir()
    _write_hrrr_with_gap(
        data_dir / "hrrr" / "hrrr_20240101.nc", datetime(year, 1, 1), n_hours, missing_hours
    )
    return create_dataloaders(
        data_dir=data_dir,
        year=year,
        back_hrs=BACK_HRS,
        lead_times=LEAD_TIMES,
        batch_size=8,
        val_split=0.2,
        lat_bounds=LAT_BOUNDS,
        lon_bounds=LON_BOUNDS,
        min_observations=10,
        use_hrrr=True,
    )


class TestInteriorGridGaps:
    def test_windows_crossing_a_missing_hour_are_excluded(self, synthetic_arlington_data) -> None:
        gap = 50
        bundle = _bundle(synthetic_arlington_data, missing_hours=[gap])
        chosen = set(bundle.train_loader.dataset.indices) | set(bundle.val_loader.dataset.indices)

        # Every window [i, i + back_hrs) that contains the gap hour is gone...
        crossing = {i for i in range(gap - BACK_HRS + 1, gap + 1)}
        assert not (chosen & crossing)
        # ...and exactly those: the neighbours on either side survive.
        assert (gap - BACK_HRS) in chosen and (gap + 1) in chosen
        assert bundle.dataset.usable_indices().tolist() == sorted(chosen)

    def test_every_remaining_sample_loads(self, synthetic_arlington_data) -> None:
        """The whole point: iterating both loaders must not raise mid-epoch."""
        bundle = _bundle(synthetic_arlington_data, missing_hours=[30, 31, 77])
        n = 0
        for loader in (bundle.train_loader, bundle.val_loader):
            for batch in loader:
                assert batch["ex_x"].shape[2] == bundle.dataset.grid_len
                n += batch["ex_x"].shape[0]
        assert n == len(bundle.dataset.usable_indices())

    def test_no_gap_is_a_contiguous_range(self, synthetic_arlington_data) -> None:
        bundle = _bundle(synthetic_arlington_data, missing_hours=[])
        start, end = bundle.dataset.usable_index_range()
        assert bundle.dataset.usable_indices().tolist() == list(range(start, end))
