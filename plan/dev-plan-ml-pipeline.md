# LOAF ML Pipeline Development Plan

**Goal:** Reproduce LocalizedWeather paper for Seattle/Pacific Northwest with continuous forecasting

**Based on:** [LocalizedWeather](https://github.com/Earth-Intelligence-Lab/LocalizedWeather) (MIT Earth Intelligence Lab)
**Paper:** Yang, Q., et al. (2024). [Local Off-Grid Weather Forecasting with Multi-Modal Earth Observation Data](https://arxiv.org/abs/2410.12938)

## Overview

Faithfully reproduce the LocalizedWeather architecture (GNN + ViT) on a Linux server, then connect to a Raspberry Pi sensor hub for local observations. This two-computer architecture separates ML concerns from sensor/edge concerns.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LINUX SERVER (ML)                                  │
│                                                                              │
│  ERA5 + HRRR + MADIS ──► Preprocessing ──► GNN + ViT Model ──► Forecasts   │
│         ▲                                                           │        │
│         │ hourly cron                                               ▼        │
│         │                                              Home Assistant API    │
└─────────┼───────────────────────────────────────────────────────────┼───────┘
          │                                                           │
          │ Local observations (future)                               │ REST/MQTT
          │                                                           ▼
┌─────────┴───────────────────────────────────────────────────────────────────┐
│                        RASPBERRY PI (Sensor Hub)                             │
│                                                                              │
│  Ultrasonic Anemometer ──► Data Logger ──► Upload to Server                 │
│  (+ future sensors)              │                                           │
│                                  ▼                                           │
│                           Local Storage                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Project Structure & Data Pipeline

### 1.1 Package Structure (Mirrors LocalizedWeather)

```
software/
├── loaf/                     # Main Python package
│   ├── __init__.py
│   ├── config.py             # YAML config loader (replaces Arg_Parser.py) ✅
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download/
│   │   │   ├── __init__.py
│   │   │   ├── era5.py       # ERA5 reanalysis download ✅
│   │   │   ├── hrrr.py       # HRRR forecast download (via Herbie) ✅
│   │   │   ├── iem.py        # IEM ASOS/AWOS download (no registration) ✅
│   │   │   └── madis.py      # MADIS station download (full network)
│   │   ├── loaders/
│   │   │   ├── __init__.py
│   │   │   ├── era5.py       # ERA5 PyTorch loader (from ERA5.py)
│   │   │   ├── hrrr.py       # HRRR PyTorch loader (from HRRR.py)
│   │   │   ├── madis.py      # MADIS PyTorch loader (from Madis.py)
│   │   │   ├── stations.py   # Station metadata (from MetaStation.py)
│   │   │   └── dataset.py    # Combined dataset (from MixData.py)
│   │   └── preprocessing/
│   │       ├── __init__.py
│   │       └── normalize.py  # Normalization (from Normalization/)
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── gnn/
│   │   │   ├── __init__.py
│   │   │   ├── mpnn.py       # Message-passing NN (from MPNN.py)
│   │   │   ├── internal.py   # Station-to-station (from GNN_Layer_Internal.py)
│   │   │   └── external.py   # Grid-to-station (from GNN_Layer_External.py)
│   │   ├── transformer/
│   │   │   ├── __init__.py
│   │   │   ├── vit.py        # Vision Transformer (from ViT.py)
│   │   │   └── embeddings.py # Station embeddings (from StationsEmbedding.py)
│   │   └── network.py        # Graph construction (from Network/)
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py        # Training loop
│   │   └── evaluate.py       # Metrics (from EvaluateModel.py)
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py      # Real-time inference
│   │   └── server.py         # REST API for forecasts
│   │
│   └── integration/
│       ├── __init__.py
│       ├── homeassistant.py  # Home Assistant client
│       └── sensor_hub.py     # Pi sensor data receiver
│
├── scripts/
│   ├── test_hrrr_download.py # HRRR download test ✅
│   ├── train.py              # Training entry point
│   ├── predict.py            # Inference entry point
│   └── serve.py              # Forecast API server
│
├── config/
│   └── seattle.yaml          # Seattle/PNW region config ✅
│
└── reference/
    └── LocalizedWeather/     # Cloned reference implementation ✅
```

### 1.2 Data Sources (Matching Paper)

| Source | Resolution | Purpose | Access |
|--------|------------|---------|--------|
| **ERA5** | 31 km | Global reanalysis baseline | CDS API (free registration) |
| **HRRR** | 3 km | High-res regional forecasts | AWS S3 (no auth) |
| **MADIS** | Point | Ground truth observations | NOAA (free registration) |

**Seattle Region:**
- Lat: 46.5°N to 49.0°N
- Lon: -124.0°W to -121.0°W

**Variables (all sources):**
- U/V wind components at 10m
- Temperature at 2m
- Dewpoint at 2m

### 1.3 ERA5 Data Pipeline

**Library:** `cdsapi` (Copernicus Climate Data Store)

**Registration:** https://cds.climate.copernicus.eu/

```python
import cdsapi

client = cdsapi.Client()
client.retrieve('reanalysis-era5-single-levels', {
    'product_type': 'reanalysis',
    'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind',
                 '2m_temperature', '2m_dewpoint_temperature'],
    'year': '2024',
    'month': '01',
    'day': ['01', '02', ...],
    'time': ['00:00', '01:00', ...],
    'area': [49.0, -124.0, 46.5, -121.0],  # N, W, S, E
    'format': 'netcdf',
}, 'era5_seattle.nc')
```

### 1.4 HRRR Data Pipeline

**Library:** `herbie-data` (downloads from AWS S3, no auth required)

```python
from herbie import Herbie

H = Herbie("2026-01-30 12:00", model="hrrr", product="sfc", fxx=6)
ds = H.xarray(":[UV]GRD:10 m above ground|:TMP:2 m|:DPT:2 m")
```

### 1.5 Station Observation Data

#### Option A: Iowa Environmental Mesonet (IEM) - No Registration Required

**Recommended for initial development.** IEM provides ASOS/AWOS data with no registration, instant access.

**Website:** https://mesonet.agron.iastate.edu/

```python
import requests

url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
params = {
    "station": "SEA",  # Seattle-Tacoma International
    "data": "tmpc,dwpc,sknt,drct",  # temp, dewpoint, wind speed/dir
    "year1": 2024, "month1": 10, "day1": 1,
    "year2": 2024, "month2": 12, "day2": 31,
    "tz": "UTC",
    "format": "onlycomma",
    "latlon": "yes",
}
response = requests.get(url, params=params)
```

**PNW ASOS/AWOS Stations:** SEA, PDX, BLI, GEG, OLM, BFI, PAE, SFF, RNT, TIW, etc. (~30-50 stations)

**Variables available:**
- `tmpc` - Temperature (°C)
- `dwpc` - Dewpoint (°C)
- `sknt` - Wind speed (knots)
- `drct` - Wind direction (degrees)
- `p01i` - 1-hour precipitation (inches)

#### Option B: MADIS - Full Station Network (Requires Registration)

**Registration:** https://madis.ncep.noaa.gov/data_application.shtml (free, 1-2 days)

**Data format:** NetCDF with quality control flags

**Key processing (from LocalizedWeather `Madis.py`):**
```python
# Quality control checks - preserve these exactly
wind_speed_check = ((data.windSpeedDD == b'S') | (data.windSpeedDD == b'V'))
wind_direction_check = ((data.windDirDD == b'S') | (data.windDirDD == b'V'))
temperature_check = ((data.temperatureDD == b'S') | (data.temperatureDD == b'V'))
wind_speed_amplitude_check = (data.windSpeed < 50)  # Filter outliers
```

**PNW Station Networks:** WSDOT, ODOT, UW, CWOP, ASOS, AWOS (~100+ stations)

#### Comparison

| Feature | IEM | MADIS |
|---------|-----|-------|
| Registration | None | 1-2 days |
| Station count (PNW) | ~30-50 | ~100+ |
| Networks | ASOS/AWOS | All networks |
| QC flags | Basic | Detailed |
| Best for | Prototyping, testing | Production training |

---

## Phase 2: Model Architecture (Matching Paper)

### 2.1 Full Architecture: GNN + ViT

The paper uses a **heterogeneous message-passing neural network (MPNN)** combined with a **Vision Transformer (ViT)**. We will implement both.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Model Architecture                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  MADIS Stations                         ERA5/HRRR Grids                     │
│       │                                       │                              │
│       ▼                                       ▼                              │
│  ┌─────────────┐                       ┌─────────────┐                      │
│  │  Station    │                       │    Grid     │                      │
│  │  Embedding  │                       │  Embedding  │                      │
│  └──────┬──────┘                       └──────┬──────┘                      │
│         │                                     │                              │
│         ▼                                     ▼                              │
│  ┌─────────────────────────────────────────────────────┐                    │
│  │              Heterogeneous Graph Network             │                    │
│  │  ┌─────────────────┐    ┌─────────────────┐         │                    │
│  │  │ Internal Layer  │    │ External Layer  │         │                    │
│  │  │ (station↔station)│    │ (grid→station) │         │                    │
│  │  └─────────────────┘    └─────────────────┘         │                    │
│  └──────────────────────────┬──────────────────────────┘                    │
│                             │                                                │
│                             ▼                                                │
│                    ┌─────────────────┐                                      │
│                    │ Vision Trans-   │                                      │
│                    │ former (ViT)    │                                      │
│                    │ Self-Attention  │                                      │
│                    └────────┬────────┘                                      │
│                             │                                                │
│                             ▼                                                │
│                    ┌─────────────────┐                                      │
│                    │  Output Head    │                                      │
│                    │  (MLP Decoder)  │                                      │
│                    └────────┬────────┘                                      │
│                             │                                                │
│                             ▼                                                │
│                   Wind Predictions                                          │
│                   (u, v at lead times)                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Model Components (from LocalizedWeather)

| Component | Source File | Purpose |
|-----------|-------------|---------|
| **MPNN** | `Modules/GNN/MPNN.py` | Main GNN orchestrator |
| **Internal Layer** | `Modules/GNN/GNN_Layer_Internal.py` | Station-to-station message passing |
| **External Layer** | `Modules/GNN/GNN_Layer_External.py` | Grid-to-station message passing |
| **ViT** | `Modules/Transformer/ViT.py` | Self-attention over tokens |
| **Station Embedding** | `Modules/Transformer/StationsEmbedding.py` | Time series → embedding |
| **Network Builder** | `Network/` | KNN/Delaunay graph construction |

### 2.3 Model Specifications

| Parameter | Value | Notes |
|-----------|-------|-------|
| hidden_dim | 128 | Match paper |
| num_gnn_layers | 2 | Internal + External |
| num_transformer_layers | 5 | Match paper |
| num_heads | 3 | Match paper |
| n_stations | ~50-100 | Seattle region |
| back_hrs | 24 | Historical window |
| lead_hrs | 48 | Forecast horizon |

**Dependencies:**
```
torch>=2.0.0
torch_geometric>=2.4.0   # For GNN layers
torch_scatter            # For message passing
```

### 2.4 Key Files to Adapt

| LocalizedWeather | LOAF Target | Adaptation |
|------------------|-------------|------------|
| `MPNN.py` | `loaf/model/gnn/mpnn.py` | Direct port |
| `GNN_Layer_Internal.py` | `loaf/model/gnn/internal.py` | Direct port |
| `GNN_Layer_External.py` | `loaf/model/gnn/external.py` | Direct port |
| `ViT.py` | `loaf/model/transformer/vit.py` | Direct port |
| `StationsEmbedding.py` | `loaf/model/transformer/embeddings.py` | Direct port |
| `Network/` | `loaf/model/network.py` | Simplify, combine |

### 2.5 Model Inputs/Outputs

**Inputs:**
- ERA5 grid: `(batch, time=25, lat, lon, vars=4)` - u10, v10, t2m, d2m
- HRRR grid: `(batch, time=25, lat, lon, vars=4)` - u10, v10, t2m, d2m
- MADIS stations: `(batch, time=24, n_stations, vars=4)` - u, v, temp, dewpoint
- Station graph: adjacency matrix from KNN/Delaunay
- Target location: `(lat, lon)`

**Outputs:**
- Wind forecast: `(batch, lead_times=8, vars=2)` for 6h, 12h, 18h, 24h, 30h, 36h, 42h, 48h

---

## Phase 3: Training Pipeline

### 3.1 Data Requirements

| Scope | Duration | Storage | Samples |
|-------|----------|---------|---------|
| Demo | 1-3 months | ~5-10 GB | ~2000 |
| Full | 5 years | ~100+ GB | ~40000 |

**Recommended:** Start with 3 months (Oct-Dec 2024) for initial validation.

### 3.2 Configuration

```yaml
# config/seattle.yaml
region:
  name: "seattle"
  lat_min: 46.5
  lat_max: 49.0
  lon_min: -124.0
  lon_max: -121.0

model:
  hidden_dim: 64
  num_layers: 3
  num_heads: 2

training:
  epochs: 100
  batch_size: 64
  learning_rate: 1e-4
  weight_decay: 1e-4

data:
  back_hrs: 24
  lead_hrs: 24
  madis_vars: ["u", "v", "temp"]
  hrrr_vars: ["u10", "v10", "u80", "v80"]
```

### 3.3 Training Script

```bash
# Train on workstation (not Pi)
python scripts/train.py --config config/seattle.yaml --epochs 100
```

**Loss function:** MSE on wind components (u, v)
**Optimizer:** AdamW with weight decay
**Validation:** 15% holdout, early stopping on val loss

---

## Phase 4: Two-Computer Deployment

### 4.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LINUX SERVER (this machine)                               │
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ Data        │    │ Model       │    │ Inference   │    │ Forecast    │  │
│  │ Download    │───►│ Training    │───►│ Service     │───►│ API         │  │
│  │ (cron)      │    │ (GPU)       │    │ (hourly)    │    │ (REST)      │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘  │
│         │                                                        │          │
│         │ ERA5, HRRR, MADIS                                      │          │
│         ▼                                                        ▼          │
│  ┌─────────────┐                                         Home Assistant     │
│  │ Data Store  │                                                            │
│  │ (NFS/local) │◄─────────────────────────────────────────────────┐        │
│  └─────────────┘                                                  │        │
└───────────────────────────────────────────────────────────────────┼────────┘
                                                                    │
                                          Local sensor observations │
                                                                    │
┌───────────────────────────────────────────────────────────────────┼────────┐
│                    RASPBERRY PI (Sensor Hub)                      │        │
│                                                                   │        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐           │        │
│  │ Ultrasonic  │───►│ Data        │───►│ Upload      │───────────┘        │
│  │ Anemometer  │    │ Logger      │    │ Service     │                    │
│  └─────────────┘    └─────────────┘    └─────────────┘                    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Server Requirements

| Resource | Requirement | Notes |
|----------|-------------|-------|
| GPU | NVIDIA with 8GB+ VRAM | For training |
| RAM | 32GB+ | Data loading |
| Storage | 500GB+ | Multi-year data archive |
| Network | Stable internet | Continuous data downloads |

### 4.3 Server Inference Pipeline

```
[Hourly Cron]
     │
     ▼
[Download latest HRRR] ─► ~100 MB/hour
     │
     ▼
[Fetch MADIS observations]
     │
     ▼
[Load ERA5 cache]
     │
     ▼
[Preprocess & Build Graph]
     │
     ▼
[PyTorch Inference] ─► ~1-5 seconds (GPU) or ~30s (CPU)
     │
     ▼
[Store Forecast] ─► PostgreSQL/SQLite
     │
     ▼
[Push to Home Assistant]
```

### 4.4 Future: Edge Deployment Options

Once the server pipeline works, we can explore:
1. **Model distillation** - Train smaller student model
2. **ONNX export** - Remove torch_geometric dependency
3. **ViT-only variant** - Skip GNN for simpler deployment
4. **Split inference** - GNN on server, final prediction on Pi

---

## Phase 5: Home Assistant Integration

### 5.1 REST API Approach (Recommended)

Simple Flask server on the Pi:

```python
# loaf/integration/homeassistant.py
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/forecast')
def get_forecast():
    forecast = load_latest_forecast()
    return jsonify({
        "wind_speed": forecast["current"]["speed"],
        "wind_direction": forecast["current"]["direction"],
        "wind_speed_unit": "m/s",
        "forecasts": [
            {"hour": 6, "speed": forecast["6h"]["speed"], "direction": forecast["6h"]["direction"]},
            {"hour": 12, "speed": forecast["12h"]["speed"], "direction": forecast["12h"]["direction"]},
            {"hour": 24, "speed": forecast["24h"]["speed"], "direction": forecast["24h"]["direction"]},
        ]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 5.2 Home Assistant Configuration

```yaml
# configuration.yaml
sensor:
  - platform: rest
    name: loaf_wind
    resource: http://raspberry-pi.local:5000/api/forecast
    json_attributes:
      - wind_speed
      - wind_direction
      - forecasts
    value_template: "{{ value_json.wind_speed }}"
    unit_of_measurement: "m/s"
    scan_interval: 3600  # Update hourly

template:
  - sensor:
      - name: "LOAF Wind Speed 6h"
        unit_of_measurement: "m/s"
        state: "{{ state_attr('sensor.loaf_wind', 'forecasts')[0].speed }}"
      - name: "LOAF Wind Direction 6h"
        unit_of_measurement: "°"
        state: "{{ state_attr('sensor.loaf_wind', 'forecasts')[0].direction }}"
```

### 5.3 Future Enhancement: Full Weather Entity

For native Home Assistant weather card support:

```python
from homeassistant.components.weather import WeatherEntity, WeatherEntityFeature

class LOAFWeatherEntity(WeatherEntity):
    _attr_supported_features = WeatherEntityFeature.FORECAST_HOURLY

    @property
    def native_wind_speed(self) -> float:
        return self._wind_speed

    async def async_forecast_hourly(self) -> list:
        return self._forecasts
```

---

## Implementation Milestones

### Milestone 1: Data Pipeline
- [x] Create package structure with `pyproject.toml`
- [x] Register for ERA5/CDS access (https://cds.climate.copernicus.eu/)
- [ ] Register for MADIS access (https://madis.ncep.noaa.gov/data_application.shtml)
- [x] Implement `loaf/data/download/era5.py`
- [x] Implement `loaf/data/download/hrrr.py` (adapt from LocalizedWeather)
- [x] Implement `loaf/data/download/iem.py` (ASOS/AWOS, no registration needed)
- [ ] Implement `loaf/data/download/madis.py` (when registration approved)
- [x] Implement data loaders (`loaf/data/loaders/`)
- [x] Implement preprocessing/normalization
- [ ] **Verify:** Download 1 month of aligned ERA5 + HRRR + IEM/MADIS data

### Milestone 2: Model Architecture
- [x] Port `MPNN.py` → `loaf/model/gnn/mpnn.py`
- [x] Port `GNN_Layer_Internal.py` → `loaf/model/gnn/internal.py`
- [x] Port `GNN_Layer_External.py` → `loaf/model/gnn/external.py`
- [x] Port `ViT.py` → `loaf/model/transformer/vit.py`
- [x] Port `StationsEmbedding.py` → `loaf/model/transformer/embeddings.py`
- [x] Implement graph construction (`loaf/model/network.py`)
- [x] **Verify:** Model forward pass works with sample data

### Milestone 3: Training Pipeline
- [x] Create PyTorch Dataset class (`loaf/data/loaders/dataset.py`)
- [x] Implement training loop (`loaf/training/trainer.py`)
- [x] Implement evaluation metrics (`loaf/training/evaluate.py`)
- [x] Write `scripts/train.py` CLI
- [ ] **Verify:** Train on 1 month data, loss decreases, metrics improve

### Milestone 4: Inference & API
- [ ] Implement predictor (`loaf/inference/predictor.py`)
- [ ] Implement REST API server (`loaf/inference/server.py`)
- [ ] Write `scripts/serve.py`
- [ ] **Verify:** API returns valid forecasts for Seattle coordinates

### Milestone 5: Home Assistant Integration
- [ ] Implement Home Assistant client (`loaf/integration/homeassistant.py`)
- [ ] Create weather entity or REST sensor configuration
- [ ] **Verify:** Wind forecast visible in Home Assistant dashboard

### Milestone 6: Continuous Operation
- [ ] Create systemd services for data download and inference
- [ ] Add cron scheduling (hourly HRRR, daily ERA5)
- [ ] Add error handling, retry logic, and alerting
- [ ] **Verify:** Runs unattended for 72+ hours

### Milestone 7: Pi Sensor Hub (Future)
- [ ] Implement sensor data logger on Pi
- [ ] Implement upload service to server
- [ ] Integrate local observations into model
- [ ] **Verify:** Local sensor data improves forecast accuracy

---

## Dependencies

```toml
# pyproject.toml
[project]
name = "loaf"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    # ML Framework
    "torch>=2.0.0",
    "torch_geometric>=2.4.0",
    "torch_scatter",
    "torch_sparse",

    # Data Access
    "herbie-data>=2024.0.0",
    "cdsapi>=0.6.0",              # ERA5 download
    "xarray>=2024.0.0",
    "netCDF4>=1.6.0",
    "cfgrib>=0.9.0",              # GRIB file support

    # Data Processing
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "scikit-learn>=1.3.0",

    # Graph Construction
    "networkx>=3.0",

    # API & Integration
    "flask>=3.0.0",
    "requests>=2.31.0",
    "pyyaml>=6.0",

    # Utilities
    "tqdm>=4.65.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "mypy>=1.0.0",
    "ruff>=0.1.0",
]
```

**Note:** `torch_geometric` and related packages require matching CUDA versions. Install with:
```bash
pip install torch_geometric torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

---

## LocalizedWeather Reference Files

Key files to study/adapt from https://github.com/Earth-Intelligence-Lab/LocalizedWeather:

### Data Download
| File | Purpose | LOAF Target |
|------|---------|-------------|
| `DataDownload/HRRR/download_hrrr.py` | HRRR via Herbie | `loaf/data/download/hrrr.py` |
| `DataDownload/ERA5/` | ERA5 download | `loaf/data/download/era5.py` |
| `DataDownload/MADIS/` | MADIS download | `loaf/data/download/madis.py` |

### Data Loading
| File | Purpose | LOAF Target |
|------|---------|-------------|
| `Dataloader/HRRR.py` | HRRR tensor loading | `loaf/data/loaders/hrrr.py` |
| `Dataloader/ERA5.py` | ERA5 tensor loading | `loaf/data/loaders/era5.py` |
| `Dataloader/Madis.py` | MADIS + QC flags | `loaf/data/loaders/madis.py` |
| `Dataloader/MetaStation.py` | Station metadata | `loaf/data/loaders/stations.py` |
| `Dataloader/MixData.py` | Combined dataset | `loaf/data/loaders/dataset.py` |
| `Normalization/` | Min-max normalization | `loaf/data/preprocessing/normalize.py` |

### Model Architecture
| File | Purpose | LOAF Target |
|------|---------|-------------|
| `Modules/GNN/MPNN.py` | Main GNN class | `loaf/model/gnn/mpnn.py` |
| `Modules/GNN/GNN_Layer_Internal.py` | Station↔Station | `loaf/model/gnn/internal.py` |
| `Modules/GNN/GNN_Layer_External.py` | Grid→Station | `loaf/model/gnn/external.py` |
| `Modules/Transformer/ViT.py` | Vision Transformer | `loaf/model/transformer/vit.py` |
| `Modules/Transformer/StationsEmbedding.py` | Time series encoder | `loaf/model/transformer/embeddings.py` |
| `Network/` | Graph construction | `loaf/model/network.py` |

### Training
| File | Purpose | LOAF Target |
|------|---------|-------------|
| `Main.py` | Training orchestration | `scripts/train.py` |
| `EvaluateModel.py` | Loss & metrics | `loaf/training/evaluate.py` |
| `Arg_Parser.py` | Config parsing | `loaf/config.py` (use YAML instead) |

---

## Action Items

**Immediate (before starting implementation):**

1. [x] Register for ERA5/CDS access: https://cds.climate.copernicus.eu/ (free, instant)
2. [ ] Register for MADIS data access: https://madis.ncep.noaa.gov/data_application.shtml (free, 1-2 days)
3. [X] Verify GPU availability and CUDA version on this server
4. [X] Ensure ~500GB free disk space for multi-year data archive
5. [x] Clone LocalizedWeather repo for reference: `git clone https://github.com/Earth-Intelligence-Lab/LocalizedWeather`

**Questions to resolve:**

- Target location coordinates (your specific deployment site in Seattle area)
- Training data range (recommend: 2020-2024 to match paper methodology)
- Home Assistant server IP/hostname for integration testing

**First coding task:** ~~Set up package structure and implement HRRR download (Milestone 1, step 1-2)~~ ✅ DONE

**Next coding task:** ~~Implement IEM download module (no registration needed, for prototyping)~~ ✅ DONE

**Next coding task:** ~~Implement data loaders (`loaf/data/loaders/`) for ERA5, HRRR, and IEM~~ ✅ DONE

**Next coding task:** ~~Implement training loop and evaluation metrics (Milestone 3)~~ ✅ DONE

**Next coding task:** Verify training with real data, then implement inference pipeline (Milestone 4)
