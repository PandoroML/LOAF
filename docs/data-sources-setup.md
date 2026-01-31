# Data Sources Setup Guide

This guide covers how to set up access to the weather data sources used by LOAF for training and inference.

## Overview

LOAF uses three primary data sources:

| Source | Resolution | Purpose | Registration |
|--------|------------|---------|--------------|
| **ERA5** | 31 km | Global reanalysis baseline | Free (instant) |
| **HRRR** | 3 km | High-res regional forecasts | None required |
| **MADIS** | Point | Ground truth observations | Free (1-2 days) |

---

## ERA5 (Copernicus Climate Data Store)

ERA5 is the ECMWF's fifth-generation atmospheric reanalysis, providing hourly data on many atmospheric, land, and oceanic climate variables from 1940 to present.

### Step 1: Create an Account

1. Go to https://cds.climate.copernicus.eu/
2. Click **"Login/Register"** in the top right
3. Complete the registration form
4. Verify your email address

### Step 2: Get Your API Key

1. Log in to https://cds.climate.copernicus.eu/
2. Click your username in the top right corner
3. Select **"Profile"** from the dropdown
4. Scroll down to find your **API Key** section
5. Copy the UID and API Key values

### Step 3: Create the Configuration File

Create the file `~/.cdsapirc` with your credentials:

```bash
cat > ~/.cdsapirc << 'EOF'
url: https://cds.climate.copernicus.eu/api
key: YOUR_UID:YOUR_API_KEY
EOF
```

Replace `YOUR_UID:YOUR_API_KEY` with your actual credentials. The format should look like:
```
key: 123456:abcd1234-5678-90ef-ghij-klmnopqrstuv
```

### Step 4: Secure the File

```bash
chmod 600 ~/.cdsapirc
```

### Step 5: Accept the Dataset License

Before downloading ERA5 data, you must accept the license terms:

1. Visit https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
2. Scroll to the bottom of the page
3. Click the **"Terms of use"** tab or find the license section
4. Click **"Accept"** to agree to the license terms

### Step 6: Verify Setup

Test your configuration:

```bash
# Run the integration test
uv run pytest software/tests/test_era5_download.py::TestIntegration -v

# Or download a small test file
uv run python -m loaf.data.download.era5 --year 2024 --month 12 -o data/era5
```

### Troubleshooting

**Error: "Missing/incomplete configuration file"**
- Ensure `~/.cdsapirc` exists and has the correct format
- Check that there are no extra spaces or characters

**Error: "403 Forbidden - required licences not accepted"**
- Visit the dataset page and accept the license terms
- Make sure you're logged in when accepting

**Error: "401 Unauthorized"**
- Verify your API key is correct
- Regenerate your API key from your profile page if needed

### Usage

```bash
# Download a single month
uv run python -m loaf.data.download.era5 --year 2024 --month 10

# Download a full year
uv run python -m loaf.data.download.era5 --year 2024

# Download a date range
uv run python -m loaf.data.download.era5 --start-year 2024 --start-month 10 --end-year 2024 --end-month 12

# Custom region (default is Seattle/PNW)
uv run python -m loaf.data.download.era5 --year 2024 --month 12 \
    --lat-min 46.5 --lat-max 49.0 \
    --lon-min -124.0 --lon-max -121.0
```

---

## HRRR (High-Resolution Rapid Refresh)

HRRR is NOAA's hourly-updating atmospheric model with 3km resolution over CONUS. Data is freely available on AWS S3 with no authentication required.

### Setup

No registration or API keys needed. HRRR data is accessed via the `herbie-data` library which downloads directly from AWS S3.

### Usage

```bash
# Download a single day
uv run python -m loaf.data.download.hrrr --date 2024-12-15

# Download a date range
uv run python -m loaf.data.download.hrrr --start-date 2024-12-01 --end-date 2024-12-31

# Custom region
uv run python -m loaf.data.download.hrrr --date 2024-12-15 \
    --lat-min 46.5 --lat-max 49.0 \
    --lon-min -124.0 --lon-max -121.0
```

### Data Availability

- HRRR data is typically available within 1-2 hours of the model run
- Historical data is available from 2014 to present
- Each file is approximately 50-100 MB depending on variables selected

---

## MADIS (Meteorological Assimilation Data Ingest System)

MADIS provides quality-controlled surface observations from multiple networks including ASOS, AWOS, mesonet stations, and citizen weather observers.

### Step 1: Request Access

1. Go to https://madis.ncep.noaa.gov/data_application.shtml
2. Fill out the data access request form
3. Wait for approval (typically 1-2 business days)
4. You'll receive credentials via email

### Step 2: Configure Credentials

*(Documentation to be added once MADIS module is implemented)*

### Data Networks in PNW Region

- **WSDOT** - Washington State DOT weather stations
- **ODOT** - Oregon DOT weather stations
- **CWOP** - Citizen Weather Observer Program
- **ASOS/AWOS** - Airport weather stations
- **UW** - University of Washington stations

---

## Data Storage Requirements

| Scope | Duration | Approximate Storage |
|-------|----------|---------------------|
| Demo/Testing | 1-3 months | 5-10 GB |
| Training (recommended) | 3-5 years | 100-200 GB |
| Full Archive | 5+ years | 300+ GB |

### Recommended Directory Structure

```
data/
├── era5/
│   ├── era5_2024_01.nc
│   ├── era5_2024_02.nc
│   └── ...
├── hrrr/
│   ├── hrrr_20241201.nc
│   ├── hrrr_20241202.nc
│   └── ...
└── madis/
    ├── madis_20241201.nc
    └── ...
```

---

## Quick Start: Download 3 Months of Data

For initial testing and validation, download October-December 2024:

```bash
# ERA5 (requires CDS setup above)
uv run python -m loaf.data.download.era5 \
    --start-year 2024 --start-month 10 \
    --end-year 2024 --end-month 12 \
    -o data/era5

# HRRR (no setup required)
uv run python -m loaf.data.download.hrrr \
    --start-date 2024-10-01 --end-date 2024-12-31 \
    -o data/hrrr
```

This will download approximately 5-10 GB of data suitable for initial model training experiments.
