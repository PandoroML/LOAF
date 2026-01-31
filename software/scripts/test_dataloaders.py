#!/usr/bin/env python3
"""Integration test for data loaders.

Downloads minimal sample data and tests all loaders end-to-end.

Usage:
    python scripts/test_dataloaders.py [--skip-download]
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add software directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_sample_data(data_dir: Path) -> bool:
    """Download minimal sample data for testing."""
    from loaf.data.download.hrrr import download_hrrr_daily
    from loaf.data.download.iem import download_iem_stations, save_iem_data, convert_to_model_units

    print("\n" + "=" * 60)
    print("STEP 1: Downloading sample data")
    print("=" * 60)

    # Create directories
    hrrr_dir = data_dir / "hrrr"
    iem_dir = data_dir / "iem"
    hrrr_dir.mkdir(parents=True, exist_ok=True)
    iem_dir.mkdir(parents=True, exist_ok=True)

    # Download 1 day of HRRR (just a few hours to save time)
    print("\n[HRRR] Downloading 1 day of HRRR data (this may take a few minutes)...")
    test_date = datetime(2024, 10, 1)
    hrrr_file = hrrr_dir / f"hrrr_{test_date.strftime('%Y%m%d')}.nc"

    if hrrr_file.exists():
        print(f"  [SKIP] {hrrr_file} already exists")
    else:
        try:
            # Download with reduced lead hours to speed up
            dataset = download_hrrr_daily(
                test_date,
                max_lead_hr=3,  # Only 3 hours of forecasts
                verbose=True,
            )
            if dataset is not None:
                dataset.to_netcdf(hrrr_file)
                print(f"  [OK] Saved to {hrrr_file}")
            else:
                print("  [WARN] No HRRR data returned")
        except Exception as e:
            print(f"  [ERROR] Failed to download HRRR: {e}")
            return False

    # Download IEM data for a few stations over 1 week
    print("\n[IEM] Downloading IEM station data...")
    iem_file = iem_dir / "iem_2024_10.parquet"

    if iem_file.exists():
        print(f"  [SKIP] {iem_file} already exists")
    else:
        try:
            # Just download a few major stations for testing
            test_stations = ["SEA", "PDX", "BLI", "OLM", "BFI"]
            start_date = datetime(2024, 10, 1)
            end_date = datetime(2024, 10, 7)  # Just 1 week

            df = download_iem_stations(
                test_stations,
                start_date,
                end_date,
            )

            if df is not None and not df.empty:
                df = convert_to_model_units(df)
                save_iem_data(df, iem_file, format="parquet")
                print(f"  [OK] Saved {len(df)} observations to {iem_file}")
            else:
                print("  [WARN] No IEM data returned")
        except Exception as e:
            print(f"  [ERROR] Failed to download IEM: {e}")
            return False

    return True


def test_hrrr_loader(data_dir: Path) -> bool:
    """Test the HRRR loader."""
    from loaf.data.loaders import HRRRLoader

    print("\n" + "=" * 60)
    print("STEP 2: Testing HRRR Loader")
    print("=" * 60)

    hrrr_dir = data_dir / "hrrr"

    try:
        loader = HRRRLoader(hrrr_dir, years=[2024])
        print(f"  [OK] Loaded HRRR data")
        print(f"       - Times: {len(loader.times)} initialization times")
        print(f"       - Steps: {loader.steps}")
        print(f"       - Nodes: {loader.n_nodes}")

        # Test sample extraction
        times = loader.times
        if len(times) >= 2:
            sample = loader.get_sample(times[0], times[1])
            print(f"  [OK] Sample extraction works")
            for var, tensor in sample.items():
                print(f"       - {var}: shape {tuple(tensor.shape)}")

        # Test statistics
        stats = loader.compute_statistics()
        print(f"  [OK] Statistics computed for {len(stats)} variables")
        for var, s in stats.items():
            print(f"       - {var}: min={s['min']:.2f}, max={s['max']:.2f}")

        return True

    except Exception as e:
        print(f"  [ERROR] HRRR loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_iem_loader(data_dir: Path) -> bool:
    """Test the IEM loader."""
    from loaf.data.loaders import IEMLoader

    print("\n" + "=" * 60)
    print("STEP 3: Testing IEM Loader")
    print("=" * 60)

    iem_dir = data_dir / "iem"

    try:
        loader = IEMLoader(iem_dir, year=2024)
        print(f"  [OK] Loaded IEM data")
        print(f"       - Stations: {loader.n_stations}")
        print(f"       - Times: {len(loader.times)}")

        # Test sample extraction
        times = loader.times
        if len(times) >= 24:
            sample = loader.get_sample(times[0], times[23])
            print(f"  [OK] Sample extraction works")
            for var, tensor in sample.items():
                if not var.endswith("_is_real"):
                    print(f"       - {var}: shape {tuple(tensor.shape)}")

        # Test statistics
        stats = loader.compute_statistics()
        print(f"  [OK] Statistics computed for {len(stats)} variables")
        for var, s in stats.items():
            print(f"       - {var}: min={s['min']:.2f}, max={s['max']:.2f}, mean={s['mean']:.2f}")

        return True

    except Exception as e:
        print(f"  [ERROR] IEM loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_station_metadata(data_dir: Path) -> bool:
    """Test station metadata and graph construction."""
    from loaf.data.loaders import StationMetadata

    print("\n" + "=" * 60)
    print("STEP 4: Testing Station Metadata")
    print("=" * 60)

    iem_dir = data_dir / "iem"

    try:
        metadata = StationMetadata.from_iem_data(
            iem_dir,
            min_observations=10,  # Low threshold for test data
        )
        print(f"  [OK] Loaded station metadata")
        print(f"       - Stations: {metadata.n_stations}")
        print(f"       - Positions shape: {tuple(metadata.positions.shape)}")

        # Test graph construction
        k_edges = metadata.get_k_edge_index(k=3)
        print(f"  [OK] KNN graph constructed")
        print(f"       - Edge index shape: {tuple(k_edges.shape)}")
        print(f"       - Number of edges: {k_edges.shape[1]}")

        if metadata.n_stations >= 4:
            delaunay_edges = metadata.get_delaunay_edge_index()
            print(f"  [OK] Delaunay graph constructed")
            print(f"       - Edge index shape: {tuple(delaunay_edges.shape)}")

        return True

    except Exception as e:
        print(f"  [ERROR] Station metadata test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_normalizers() -> bool:
    """Test normalization utilities."""
    from loaf.data.preprocessing import MinMaxNormalizer, StandardNormalizer, NormalizerCollection
    import torch

    print("\n" + "=" * 60)
    print("STEP 5: Testing Normalizers")
    print("=" * 60)

    try:
        # Test MinMaxNormalizer
        normalizer = MinMaxNormalizer(0.0, 100.0)
        x = torch.tensor([25.0, 50.0, 75.0])
        encoded = normalizer.encode(x)
        decoded = normalizer.decode(encoded)

        assert torch.allclose(x, decoded, atol=1e-4), "MinMax encode/decode failed"
        print(f"  [OK] MinMaxNormalizer: {x.tolist()} -> {encoded.tolist()} -> {decoded.tolist()}")

        # Test StandardNormalizer
        normalizer = StandardNormalizer(50.0, 25.0)
        encoded = normalizer.encode(x)
        decoded = normalizer.decode(encoded)

        assert torch.allclose(x, decoded, atol=1e-4), "Standard encode/decode failed"
        print(f"  [OK] StandardNormalizer works")

        # Test NormalizerCollection
        collection = NormalizerCollection(normalizer_type="minmax")
        collection.fit_from_stats({
            "temp": {"min": -10, "max": 40, "mean": 15, "std": 10},
            "wind": {"min": 0, "max": 50, "mean": 10, "std": 8},
        })

        temp = torch.tensor([0.0, 15.0, 30.0])
        encoded_temp = collection.encode("temp", temp)
        print(f"  [OK] NormalizerCollection: temp {temp.tolist()} -> {encoded_temp.tolist()}")

        return True

    except Exception as e:
        print(f"  [ERROR] Normalizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_combined_dataset(data_dir: Path) -> bool:
    """Test the combined WeatherDataset."""
    from loaf.data.loaders import WeatherDataset, StationMetadata, IEMLoader, HRRRLoader

    print("\n" + "=" * 60)
    print("STEP 6: Testing Combined Dataset")
    print("=" * 60)

    try:
        # Load components
        iem_dir = data_dir / "iem"
        hrrr_dir = data_dir / "hrrr"

        print("  Loading station metadata...")
        station_metadata = StationMetadata.from_iem_data(
            iem_dir,
            min_observations=10,
        )

        print("  Loading IEM observations...")
        station_loader = IEMLoader(iem_dir, year=2024, station_metadata=station_metadata)

        print("  Loading HRRR grid data...")
        grid_loader = HRRRLoader(hrrr_dir, years=[2024])

        # Test 1: Station-only dataset (no grid loader)
        print("  Creating station-only dataset...")
        dataset_station = WeatherDataset(
            year=2024,
            back_hrs=6,  # Short window for testing
            lead_hours=3,
            station_metadata=station_metadata,
            station_loader=station_loader,
            grid_loader=None,  # Skip grid for this test
        )

        print(f"  [OK] Station-only dataset created")
        print(f"       - Length: {len(dataset_station)} samples")
        print(f"       - Stations: {dataset_station.n_stations}")

        # Get a sample from station-only dataset
        if len(dataset_station) > 0:
            sample = dataset_station[0]
            print(f"  [OK] Station-only sample retrieval works")
            print(f"       Sample keys: {list(sample.keys())}")

        # Test 2: Combined dataset with grid (use index that aligns with HRRR data)
        # HRRR data is for Oct 1, IEM data starts at index 0 = Jan 1
        # October 1 is approximately day 274, so hour index ~= 274*24 = 6576
        print("  Creating combined dataset with grid...")
        dataset = WeatherDataset(
            year=2024,
            back_hrs=6,  # Short window for testing
            lead_hours=3,
            station_metadata=station_metadata,
            station_loader=station_loader,
            grid_loader=grid_loader,
        )

        print(f"  [OK] Combined dataset created")
        print(f"       - Length: {len(dataset)} samples")
        print(f"       - Stations: {dataset.n_stations}")
        print(f"       - Grid nodes: {dataset.n_grid_nodes}")

        # Get a sample at an index that aligns with October data
        # October 1 = day 274 (0-indexed from Jan 1), so start at hour 274*24 = 6576
        oct_start_idx = 274 * 24  # First hour of October 1
        if len(dataset) > oct_start_idx:
            sample = dataset[oct_start_idx]
            print(f"  [OK] Combined sample retrieval works (index {oct_start_idx})")
            print(f"       Sample keys: {list(sample.keys())}")

            for key, value in sample.items():
                if hasattr(value, "shape"):
                    print(f"       - {key}: shape {tuple(value.shape)}")

        return True

    except Exception as e:
        print(f"  [ERROR] Combined dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test data loaders")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading data (assumes data already exists)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/test"),
        help="Directory for test data (default: data/test)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.absolute()
    print(f"Test data directory: {data_dir}")

    results = {}

    # Step 1: Download data
    if not args.skip_download:
        results["download"] = download_sample_data(data_dir)
        if not results["download"]:
            print("\n[FAIL] Download failed, cannot continue with other tests")
            return 1
    else:
        print("\n[SKIP] Skipping download (--skip-download)")
        results["download"] = True

    # Step 2-4: Test individual loaders
    results["hrrr"] = test_hrrr_loader(data_dir)
    results["iem"] = test_iem_loader(data_dir)
    results["stations"] = test_station_metadata(data_dir)

    # Step 5: Test normalizers
    results["normalizers"] = test_normalizers()

    # Step 6: Test combined dataset
    results["dataset"] = test_combined_dataset(data_dir)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print("\n[FAILURE] Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
