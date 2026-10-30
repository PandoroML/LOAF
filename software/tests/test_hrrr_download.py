"""Tests for the HRRR download module."""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from loaf.data.download.hrrr import (
    DEFAULT_VARIABLES,
    SEATTLE_BOUNDS,
    _cached_file_covers_lead_hr,
    download_hrrr_hourly,
    download_hrrr_range,
)


def _make_hrrr_ds(n_steps: int) -> xr.Dataset:
    """A minimal dataset shaped like a real downloaded HRRR day - just enough
    for _cached_file_covers_lead_hr's "step" dim check and to round-trip
    through to_netcdf."""
    return xr.Dataset(
        {"u10": (["step", "y", "x"], np.zeros((n_steps, 2, 2), dtype="float32"))},
        coords={
            "step": np.arange(n_steps),
            "latitude": (["y", "x"], np.zeros((2, 2))),
            "longitude": (["y", "x"], np.zeros((2, 2))),
            "time": datetime(2024, 10, 1),
        },
    )


class TestConstants:
    """Tests for module constants."""

    def test_seattle_bounds_values(self) -> None:
        """Test Seattle bounds are correct for PNW region."""
        assert SEATTLE_BOUNDS["lat_min"] == 46.5
        assert SEATTLE_BOUNDS["lat_max"] == 49.0
        # HRRR uses 0-360 longitude
        assert SEATTLE_BOUNDS["lon_min"] == -124.0 + 360
        assert SEATTLE_BOUNDS["lon_max"] == -121.0 + 360

    def test_seattle_bounds_valid_range(self) -> None:
        """Test that bounds define a valid geographic region."""
        assert SEATTLE_BOUNDS["lat_min"] < SEATTLE_BOUNDS["lat_max"]
        assert SEATTLE_BOUNDS["lon_min"] < SEATTLE_BOUNDS["lon_max"]

    def test_default_variables_contains_wind(self) -> None:
        """Test that default variables include wind components."""
        assert "UGRD" in DEFAULT_VARIABLES
        assert "VGRD" in DEFAULT_VARIABLES

    def test_default_variables_contains_temp(self) -> None:
        """Test that default variables include temperature."""
        assert "TMP" in DEFAULT_VARIABLES
        assert "DPT" in DEFAULT_VARIABLES


class TestDownloadHRRRHourly:
    """Tests for download_hrrr_hourly function."""

    def test_accepts_string_datetime(self) -> None:
        """Test that function accepts string datetime format."""
        with patch("loaf.data.download.hrrr.Herbie") as mock_herbie:
            mock_herbie.return_value.grib = None  # Simulate no data

            result = download_hrrr_hourly(
                "2024-01-15 12:00",
                max_lead_hr=0,
            )

            assert result is None  # No data returned

    def test_accepts_datetime_object(self) -> None:
        """Test that function accepts datetime object."""
        with patch("loaf.data.download.hrrr.Herbie") as mock_herbie:
            mock_herbie.return_value.grib = None

            result = download_hrrr_hourly(
                datetime(2024, 1, 15, 12, 0),
                max_lead_hr=0,
            )

            assert result is None

    def test_returns_none_when_no_data(self) -> None:
        """Test that function returns None when no GRIB data available."""
        with patch("loaf.data.download.hrrr.Herbie") as mock_herbie:
            mock_herbie.return_value.grib = None

            result = download_hrrr_hourly(
                "2024-01-15 12:00",
                max_lead_hr=1,
            )

            assert result is None

    def test_custom_bounds(self) -> None:
        """Test that custom lat/lon bounds are passed correctly."""
        with patch("loaf.data.download.hrrr.Herbie") as mock_herbie:
            mock_herbie.return_value.grib = None

            download_hrrr_hourly(
                "2024-01-15 12:00",
                lat_min=40.0,
                lat_max=45.0,
                lon_min=250.0,
                lon_max=260.0,
                max_lead_hr=0,
            )

            # Function should complete without error
            mock_herbie.assert_called()


class TestLongitudeConversion:
    """Tests for longitude coordinate handling."""

    def test_seattle_lon_in_hrrr_format(self) -> None:
        """Test that Seattle longitudes are in 0-360 format for HRRR."""
        # HRRR uses 0-360 longitude
        # Seattle is around -122W, which should be 238 in HRRR format
        assert 235 < SEATTLE_BOUNDS["lon_min"] < 240
        assert 235 < SEATTLE_BOUNDS["lon_max"] < 245

    def test_lon_conversion_formula(self) -> None:
        """Test the longitude conversion from -180/180 to 0/360."""
        # Standard conversion: if lon < 0, add 360
        test_lon = -122.0
        hrrr_lon = test_lon + 360

        assert hrrr_lon == 238.0
        assert 0 <= hrrr_lon <= 360


class TestCachedFileCoversLeadHr:
    """Tests for _cached_file_covers_lead_hr, the staleness check that lets
    download_hrrr_range tell a cached analysis-only (or lower max_lead_hr)
    file apart from one with enough forecast steps already archived."""

    def test_fewer_steps_than_requested_is_not_covered(self, tmp_path: Path) -> None:
        f = tmp_path / "hrrr_20241001.nc"
        _make_hrrr_ds(1).to_netcdf(f)  # analysis-only, like the old downloads

        assert _cached_file_covers_lead_hr(f, max_lead_hr=18) is False

    def test_exact_step_count_is_covered(self, tmp_path: Path) -> None:
        f = tmp_path / "hrrr_20241001.nc"
        _make_hrrr_ds(4).to_netcdf(f)  # steps 0..3, i.e. max_lead_hr=3

        assert _cached_file_covers_lead_hr(f, max_lead_hr=3) is True

    def test_more_steps_than_requested_is_covered(self, tmp_path: Path) -> None:
        f = tmp_path / "hrrr_20241001.nc"
        _make_hrrr_ds(19).to_netcdf(f)

        assert _cached_file_covers_lead_hr(f, max_lead_hr=3) is True

    def test_unreadable_file_is_not_covered(self, tmp_path: Path) -> None:
        f = tmp_path / "hrrr_20241001.nc"
        f.write_text("not a netcdf file")

        assert _cached_file_covers_lead_hr(f, max_lead_hr=0) is False


class TestDownloadHRRRRange:
    """Tests for download_hrrr_range's cache-reuse decision, mocking
    download_hrrr_daily so these don't hit the network."""

    def test_stale_cached_file_is_redownloaded(self, tmp_path: Path) -> None:
        stale = tmp_path / "hrrr_20241001.nc"
        _make_hrrr_ds(1).to_netcdf(stale)  # only step=0 archived

        with patch(
            "loaf.data.download.hrrr.download_hrrr_daily",
            return_value=_make_hrrr_ds(19),
        ) as mock_download:
            download_hrrr_range(
                datetime(2024, 10, 1), datetime(2024, 10, 1), tmp_path, max_lead_hr=18
            )

        mock_download.assert_called_once()
        assert xr.open_dataset(stale).sizes["step"] == 19

    def test_sufficiently_covered_file_is_skipped(self, tmp_path: Path) -> None:
        cached = tmp_path / "hrrr_20241001.nc"
        _make_hrrr_ds(19).to_netcdf(cached)

        with patch(
            "loaf.data.download.hrrr.download_hrrr_daily",
            return_value=_make_hrrr_ds(19),
        ) as mock_download:
            saved = download_hrrr_range(
                datetime(2024, 10, 1), datetime(2024, 10, 1), tmp_path, max_lead_hr=18
            )

        mock_download.assert_not_called()
        assert saved == [cached]

    def test_raising_max_lead_hr_forces_redownload_of_prior_cache(
        self, tmp_path: Path
    ) -> None:
        """A file downloaded at a lower max_lead_hr shouldn't be treated as
        covering a later, larger request."""
        cached = tmp_path / "hrrr_20241001.nc"
        _make_hrrr_ds(4).to_netcdf(cached)  # downloaded with max_lead_hr=3

        with patch(
            "loaf.data.download.hrrr.download_hrrr_daily",
            return_value=_make_hrrr_ds(19),
        ) as mock_download:
            download_hrrr_range(
                datetime(2024, 10, 1), datetime(2024, 10, 1), tmp_path, max_lead_hr=18
            )

        mock_download.assert_called_once()

    def test_missing_file_is_downloaded(self, tmp_path: Path) -> None:
        with patch(
            "loaf.data.download.hrrr.download_hrrr_daily",
            return_value=_make_hrrr_ds(19),
        ) as mock_download:
            saved = download_hrrr_range(
                datetime(2024, 10, 2), datetime(2024, 10, 2), tmp_path, max_lead_hr=18
            )

        mock_download.assert_called_once()
        assert saved == [tmp_path / "hrrr_20241002.nc"]

    def test_corrupt_cached_file_is_redownloaded(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "hrrr_20241001.nc"
        corrupt.write_text("not a netcdf file")

        with patch(
            "loaf.data.download.hrrr.download_hrrr_daily",
            return_value=_make_hrrr_ds(19),
        ) as mock_download:
            download_hrrr_range(
                datetime(2024, 10, 1), datetime(2024, 10, 1), tmp_path, max_lead_hr=18
            )

        mock_download.assert_called_once()
        assert xr.open_dataset(corrupt).sizes["step"] == 19


class TestIntegration:
    """Integration tests (require network access)."""

    @pytest.mark.slow
    @pytest.mark.network
    def test_actual_download(self) -> None:
        """Test actual HRRR download (slow, requires network)."""
        # This test is marked as slow and network-dependent
        # Run with: pytest -m "slow and network"
        from datetime import timedelta

        # Use yesterday's data (more reliable availability)
        test_date = datetime.now() - timedelta(days=1)
        test_time = test_date.replace(hour=12, minute=0)

        result = download_hrrr_hourly(
            test_time,
            max_lead_hr=0,  # Just analysis time
        )

        assert result is not None
        assert isinstance(result, xr.Dataset)
        assert "u10" in result.data_vars or "UGRD" in str(result.data_vars)
