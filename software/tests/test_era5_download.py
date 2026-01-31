"""Tests for the ERA5 download module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import xarray as xr
import numpy as np

from loaf.data.download.era5 import (
    DATASET,
    DEFAULT_VARIABLES,
    SEATTLE_BOUNDS,
    download_era5_month,
    download_era5_range,
    download_era5_year,
    load_era5_month,
)


class TestConstants:
    """Tests for module constants."""

    def test_seattle_bounds_values(self) -> None:
        """Test Seattle bounds are correct for PNW region."""
        assert SEATTLE_BOUNDS["lat_min"] == 46.5
        assert SEATTLE_BOUNDS["lat_max"] == 49.0
        assert SEATTLE_BOUNDS["lon_min"] == -124.0
        assert SEATTLE_BOUNDS["lon_max"] == -121.0

    def test_seattle_bounds_valid_range(self) -> None:
        """Test that bounds define a valid geographic region."""
        assert SEATTLE_BOUNDS["lat_min"] < SEATTLE_BOUNDS["lat_max"]
        assert SEATTLE_BOUNDS["lon_min"] < SEATTLE_BOUNDS["lon_max"]

    def test_seattle_bounds_use_standard_longitude(self) -> None:
        """Test that ERA5 bounds use -180/180 longitude format."""
        # ERA5 uses standard -180 to 180 longitude (unlike HRRR which uses 0-360)
        assert -180 <= SEATTLE_BOUNDS["lon_min"] <= 180
        assert -180 <= SEATTLE_BOUNDS["lon_max"] <= 180

    def test_default_variables_contains_wind(self) -> None:
        """Test that default variables include wind components."""
        assert "10m_u_component_of_wind" in DEFAULT_VARIABLES
        assert "10m_v_component_of_wind" in DEFAULT_VARIABLES

    def test_default_variables_contains_temp(self) -> None:
        """Test that default variables include temperature."""
        assert "2m_temperature" in DEFAULT_VARIABLES
        assert "2m_dewpoint_temperature" in DEFAULT_VARIABLES

    def test_default_variables_contains_solar(self) -> None:
        """Test that default variables include solar radiation."""
        assert "surface_net_solar_radiation" in DEFAULT_VARIABLES

    def test_dataset_name(self) -> None:
        """Test that the correct CDS dataset is specified."""
        assert DATASET == "reanalysis-era5-single-levels"


class TestDownloadERA5Month:
    """Tests for download_era5_month function."""

    def test_skips_existing_file(self, tmp_path: Path) -> None:
        """Test that existing files are skipped."""
        existing_file = tmp_path / "era5_2024_01.nc"
        existing_file.touch()

        result = download_era5_month(2024, 1, existing_file)

        assert result == existing_file

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        """Test that output directory is created if it doesn't exist."""
        nested_path = tmp_path / "nested" / "dir" / "era5_2024_01.nc"

        with patch("loaf.data.download.era5.cdsapi.Client") as mock_client:
            mock_client.return_value.retrieve.return_value.download.return_value = None
            download_era5_month(2024, 1, nested_path)

        assert nested_path.parent.exists()

    def test_correct_api_request_format(self, tmp_path: Path) -> None:
        """Test that the CDS API request has correct format."""
        output_file = tmp_path / "era5_2024_06.nc"

        with patch("loaf.data.download.era5.cdsapi.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            download_era5_month(2024, 6, output_file)

            # Check retrieve was called
            mock_instance.retrieve.assert_called_once()
            call_args = mock_instance.retrieve.call_args

            # First arg should be dataset name
            assert call_args[0][0] == DATASET

            # Second arg should be the request dict
            request = call_args[0][1]
            assert request["product_type"] == ["reanalysis"]
            assert request["data_format"] == "netcdf"
            assert request["year"] == ["2024"]
            assert request["month"] == ["06"]
            assert len(request["day"]) == 31
            assert len(request["time"]) == 24

    def test_area_format(self, tmp_path: Path) -> None:
        """Test that area is formatted correctly [N, W, S, E]."""
        output_file = tmp_path / "era5_2024_01.nc"

        with patch("loaf.data.download.era5.cdsapi.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            download_era5_month(
                2024, 1, output_file,
                lat_min=40.0, lat_max=50.0,
                lon_min=-125.0, lon_max=-120.0
            )

            request = mock_instance.retrieve.call_args[0][1]
            area = request["area"]

            # ERA5 format: [North, West, South, East]
            assert area == [50.0, -125.0, 40.0, -120.0]

    def test_returns_none_on_api_error(self, tmp_path: Path) -> None:
        """Test that function returns None when API fails."""
        output_file = tmp_path / "era5_2024_01.nc"

        with patch("loaf.data.download.era5.cdsapi.Client") as mock_client:
            mock_client.return_value.retrieve.side_effect = Exception("API Error")

            result = download_era5_month(2024, 1, output_file)

            assert result is None

    def test_custom_variables(self, tmp_path: Path) -> None:
        """Test that custom variables are passed to API."""
        output_file = tmp_path / "era5_2024_01.nc"
        custom_vars = ["10m_u_component_of_wind", "10m_v_component_of_wind"]

        with patch("loaf.data.download.era5.cdsapi.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            download_era5_month(2024, 1, output_file, variables=custom_vars)

            request = mock_instance.retrieve.call_args[0][1]
            assert request["variable"] == custom_vars


class TestDownloadERA5Year:
    """Tests for download_era5_year function."""

    def test_downloads_all_12_months(self, tmp_path: Path) -> None:
        """Test that all 12 months are downloaded."""
        with patch("loaf.data.download.era5.download_era5_month") as mock_download:
            mock_download.return_value = tmp_path / "test.nc"

            download_era5_year(2024, tmp_path)

            assert mock_download.call_count == 12

    def test_correct_filenames(self, tmp_path: Path) -> None:
        """Test that correct filenames are generated for each month."""
        with patch("loaf.data.download.era5.download_era5_month") as mock_download:
            mock_download.return_value = tmp_path / "test.nc"

            download_era5_year(2024, tmp_path)

            # Check first and last month filenames
            first_call = mock_download.call_args_list[0]
            assert first_call[0][0] == 2024  # year
            assert first_call[0][1] == 1     # month
            assert "era5_2024_01.nc" in str(first_call[0][2])

            last_call = mock_download.call_args_list[11]
            assert last_call[0][0] == 2024   # year
            assert last_call[0][1] == 12     # month
            assert "era5_2024_12.nc" in str(last_call[0][2])

    def test_returns_list_of_paths(self, tmp_path: Path) -> None:
        """Test that function returns list of downloaded paths."""
        with patch("loaf.data.download.era5.download_era5_month") as mock_download:
            mock_download.return_value = tmp_path / "test.nc"

            result = download_era5_year(2024, tmp_path)

            assert isinstance(result, list)
            assert len(result) == 12


class TestDownloadERA5Range:
    """Tests for download_era5_range function."""

    def test_single_month_range(self, tmp_path: Path) -> None:
        """Test downloading a single month."""
        with patch("loaf.data.download.era5.download_era5_month") as mock_download:
            mock_download.return_value = tmp_path / "test.nc"

            download_era5_range(2024, 6, 2024, 6, tmp_path)

            assert mock_download.call_count == 1

    def test_cross_year_range(self, tmp_path: Path) -> None:
        """Test downloading across year boundary."""
        with patch("loaf.data.download.era5.download_era5_month") as mock_download:
            mock_download.return_value = tmp_path / "test.nc"

            # Nov 2023 to Feb 2024 = 4 months
            download_era5_range(2023, 11, 2024, 2, tmp_path)

            assert mock_download.call_count == 4

    def test_multi_year_range(self, tmp_path: Path) -> None:
        """Test downloading multiple full years."""
        with patch("loaf.data.download.era5.download_era5_month") as mock_download:
            mock_download.return_value = tmp_path / "test.nc"

            # Full 2023 and 2024 = 24 months
            download_era5_range(2023, 1, 2024, 12, tmp_path)

            assert mock_download.call_count == 24

    def test_handles_failed_downloads(self, tmp_path: Path) -> None:
        """Test that failed downloads are excluded from results."""
        with patch("loaf.data.download.era5.download_era5_month") as mock_download:
            # First month succeeds, second fails
            mock_download.side_effect = [tmp_path / "test.nc", None]

            result = download_era5_range(2024, 1, 2024, 2, tmp_path)

            assert len(result) == 1


class TestLoadERA5Month:
    """Tests for load_era5_month function."""

    def test_renames_variables(self, tmp_path: Path) -> None:
        """Test that ERA5 variable names are renamed to standard names."""
        # Create a mock ERA5 dataset
        ds = xr.Dataset({
            "u10": (["time", "lat", "lon"], np.random.rand(24, 10, 10)),
            "v10": (["time", "lat", "lon"], np.random.rand(24, 10, 10)),
            "t2m": (["time", "lat", "lon"], np.random.rand(24, 10, 10)),
            "d2m": (["time", "lat", "lon"], np.random.rand(24, 10, 10)),
            "ssr": (["time", "lat", "lon"], np.random.rand(24, 10, 10)),
        })

        test_file = tmp_path / "test_era5.nc"
        ds.to_netcdf(test_file)

        loaded = load_era5_month(test_file)

        # Check renamed variables exist
        assert "u" in loaded.data_vars
        assert "v" in loaded.data_vars
        assert "temp" in loaded.data_vars
        assert "dewpoint" in loaded.data_vars
        assert "solar_radiation" in loaded.data_vars

        # Check original names don't exist
        assert "u10" not in loaded.data_vars
        assert "v10" not in loaded.data_vars
        assert "t2m" not in loaded.data_vars

        loaded.close()

    def test_handles_partial_variables(self, tmp_path: Path) -> None:
        """Test loading file with only some variables."""
        # Create dataset with only wind variables
        ds = xr.Dataset({
            "u10": (["time", "lat", "lon"], np.random.rand(24, 10, 10)),
            "v10": (["time", "lat", "lon"], np.random.rand(24, 10, 10)),
        })

        test_file = tmp_path / "test_era5_partial.nc"
        ds.to_netcdf(test_file)

        loaded = load_era5_month(test_file)

        assert "u" in loaded.data_vars
        assert "v" in loaded.data_vars
        assert "temp" not in loaded.data_vars

        loaded.close()


class TestIntegration:
    """Integration tests (require network access and CDS credentials)."""

    @pytest.fixture
    def cds_available(self) -> bool:
        """Check if CDS credentials are available."""
        cdsapirc = Path.home() / ".cdsapirc"
        return cdsapirc.exists()

    @pytest.mark.slow
    @pytest.mark.network
    @pytest.mark.cds
    def test_actual_download(self, tmp_path: Path, cds_available: bool) -> None:
        """Test actual ERA5 download (slow, requires network and CDS credentials).

        Run with: pytest -m "slow and network and cds"
        """
        if not cds_available:
            pytest.skip("CDS credentials not found at ~/.cdsapirc")

        # Download a single month of data (smallest practical unit)
        result = download_era5_month(
            2024, 1, tmp_path / "era5_2024_01.nc",
            # Use smaller area for faster test
            lat_min=47.5, lat_max=48.0,
            lon_min=-122.5, lon_max=-122.0,
        )

        assert result is not None
        assert result.exists()

        # Verify the downloaded data
        ds = xr.open_dataset(result)
        assert "u10" in ds.data_vars or "u" in ds.data_vars
        ds.close()
