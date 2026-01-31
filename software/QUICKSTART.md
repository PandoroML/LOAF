# LOAF Quickstart Guide

## Prerequisites

1. **uv** - Fast Python package manager: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. **ERA5 API Access** (optional but recommended): Register at https://cds.climate.copernicus.eu/

## Setup

```bash
cd /home/keenan/code/LOAF
uv sync
```

## Step 1: Download Data

### Option A: Quick Start (IEM + HRRR only, no registration)

```bash
cd software

# Download IEM station observations (no auth required)
uv run python -m loaf.data.download.iem \
    --start-date 2024-10-01 \
    --end-date 2024-10-31 \
    --output data/iem/iem_2024_10.parquet

# Download HRRR forecasts (no auth required)
uv run python -m loaf.data.download.hrrr \
    --start-date 2024-10-01 \
    --end-date 2024-10-31 \
    --output-dir data/hrrr
```

### Option B: Full Setup (includes ERA5)

First, configure ERA5 API credentials in `~/.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api
key: YOUR_UID:YOUR_API_KEY
```

Then download:
```bash
# Download ERA5 reanalysis
uv run python -m loaf.data.download.era5 \
    --year 2024 \
    --month 10 \
    --output data/era5/era5_2024_10.nc

# Download IEM + HRRR as above
```

## Step 2: Train Model

```bash
cd software

# Train GNN model (default)
uv run python scripts/train.py \
    --config config/seattle.yaml \
    --data-dir data \
    --year 2024 \
    --epochs 50 \
    --checkpoint checkpoints/

# Or train ViT model
uv run python scripts/train.py \
    --config config/seattle.yaml \
    --model vit \
    --epochs 50
```

### Training Options

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Config YAML file | `config/seattle.yaml` |
| `--model` | Model type: `gnn` or `vit` | `gnn` |
| `--epochs` | Number of epochs | 100 |
| `--batch-size` | Batch size | 64 |
| `--lr` | Learning rate | 1e-4 |
| `--checkpoint` | Directory to save models | None |
| `--resume` | Resume from checkpoint | None |
| `--device` | Device: `auto`, `cuda`, `cpu` | `auto` |

## Step 3: Verify Installation

```bash
# Test model forward pass
uv run python scripts/test_model_forward.py

# Test training loop with synthetic data
uv run python scripts/test_training.py
```

## Data Directory Structure

After downloading, your data directory should look like:
```
software/data/
├── hrrr/
│   ├── hrrr_20241001.nc
│   ├── hrrr_20241002.nc
│   └── ...
├── iem/
│   └── iem_2024_10.parquet
└── era5/
    └── era5_2024_10.nc
```

## CLI Commands

```bash
uv run loaf-download-hrrr --help    # Download HRRR data
uv run loaf-download-era5 --help    # Download ERA5 data
uv run loaf-download-iem --help     # Download IEM station data
```

## Minimal Example (1 day of data for testing)

```bash
cd software

# Download just 1 day of data for testing
uv run python -m loaf.data.download.iem \
    --start-date 2024-10-15 \
    --end-date 2024-10-15 \
    --output data/iem/test.parquet

uv run python -m loaf.data.download.hrrr \
    --start-date 2024-10-15 \
    --end-date 2024-10-15 \
    --output-dir data/hrrr

# Quick training test (3 epochs)
uv run python scripts/train.py --epochs 3 --batch-size 8
```

## Troubleshooting

- **HRRR download fails**: Data may not be available for very recent dates. Try dates 2+ days ago.
- **ERA5 slow**: ERA5 downloads are queued on the CDS server. Initial requests may take 30+ minutes.
- **Out of memory**: Reduce `--batch-size` (try 16 or 8)
- **torch_geometric not found**: Run `uv sync` to install all dependencies
