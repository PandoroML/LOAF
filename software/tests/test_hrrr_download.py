"""Tests for the HRRR download module."""

from datetime import datetime
from unittest.mock import patch

import pytest
import xarray as xr
from loaf.data.download.hrrr import (
    DEFAULT_VARIABLES,
    SEATTLE_BOUNDS,
    download_hrrr_hourly,
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
