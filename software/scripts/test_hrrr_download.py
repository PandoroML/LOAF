#!/usr/bin/env python3
"""Test script for HRRR download functionality.

Downloads a single hour of HRRR data for the Seattle region to verify
the download pipeline is working correctly.
"""

import sys
from datetime import datetime, timedelta

# Test with yesterday's data (more likely to be available)
test_date = datetime.now() - timedelta(days=1)
test_time = test_date.replace(hour=12, minute=0, second=0, microsecond=0)

print(f"Testing HRRR download for: {test_time}")
print("=" * 60)

try:
    from loaf.data.download.hrrr import download_hrrr_hourly, SEATTLE_BOUNDS, DEFAULT_VARIABLES

    print(f"Seattle bounds: {SEATTLE_BOUNDS}")
    print(f"Variables: {DEFAULT_VARIABLES}")
    print()

    # Download just 2 lead hours to test quickly
    print("Downloading 2 forecast hours (fxx=0,1)...")
    dataset = download_hrrr_hourly(
        test_time,
        var_list=DEFAULT_VARIABLES,
        max_lead_hr=1,  # Just 2 time steps (0 and 1)
        verbose=True,
    )

    if dataset is not None:
        print("\nDownload successful!")
        print(f"Dataset shape: {dict(dataset.dims)}")
        print(f"Variables: {list(dataset.data_vars)}")
        print(f"Coordinates: {list(dataset.coords)}")
        print("\nSample data preview:")
        print(dataset)
    else:
        print("\nDownload returned None - no data available")
        sys.exit(1)

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
