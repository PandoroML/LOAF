# LOAF (Local Observations and Atmospheric Forecasting)

Open source hyperlocal weather forecasting combining machine learning forecast models with local station observations.

- Do you want to improve the weather forecasts at a specific place? 
- Do you have access to local sensors or are intersted in building them? 
- Are you interested in understanding the full process and not relying on big tech companies hiding proprietary algorithms to do so? 

If you answered yes to all of the above, then LOAF is for you.

Part of [Pandoro](https://pandoro.today) — Bread Board Foundry's open science ML tools for climate and environmental research.

This is definitely a work in progress, so stay tuned!

## Planned Architecture

```
┌─────────────────┐      ┌─────────────────────────────────────┐      ┌─────────────────┐
│  DIY Sensors    │ ──── │          Raspberry Pi               │ ──── │ Home Assistant  │
│  (Anemometer,   │      │  • Data storage                     │      │                 │
│   Temperature,  │      │  • ML forecast processing           │      │  Weather entity │
│   etc.)         │      │  • Local prediction generation      │      │  integrations   │
└─────────────────┘      └─────────────────────────────────────┘      └─────────────────┘
```

**Data Flow:**
1. **Sensors → Raspberry Pi**: Local sensors connect to the Pi via a common sensors library, logging observations to local storage
2. **Forecast Processing**: The Pi runs the trained ML model, combining local sensor data with regional forecasts (HRRR/GFS) to generate hyperlocal predictions
3. **Home Assistant Integration**: Predictions are exposed as Home Assistant entities, allowing integration with automations, dashboards, and alerts

## Latest Cool Picture

The first of the ultrasonic transducers have arrived!

![IMG_9646](https://github.com/user-attachments/assets/8b167aed-2db6-44ab-8047-cf3c9de885f3)

## Current priorities:

- Set up HRRR/MADIS data pipeline
- Train regional model and validate against local observations
- Deploy the model to local hardware (raspberry pi)
- Create Home Assistant widget for local wind predictions

---

## Why LOAF?

Standard weather forecasts operate on 3km grids. That resolution can't capture the wind patterns at your specific site—whether it's a backyard wind turbine, a fire-prone hillside, or a remote research station.

Recent ML research shows that fusing gridded forecasts with local sensor data via transformers can reduce prediction error by up to 80%. Commercial services like Tomorrow.io offer this, but they're proprietary and subscription-based. The academic code exists, but there's no easy way to go from "I have a Raspberry Pi" to "I have an improved local forecast."

LOAF bridges that gap: open source hardware, open source models, no vendor lock-in. Build a sensor, train a model for your region, run inference locally.

## About

LOAF generates hyperlocal weather forecasts for locations without nearby weather stations by combining:

- Regional forecast models (NOAA GFS, HRRR, ERA5)
- Sparse local weather station observations
- Multi-modal transformer architecture for spatial-temporal fusion

**Built on research from MIT Earth Intelligence Lab:**
- GitHub: [Earth-Intelligence-Lab/LocalizedWeather](https://github.com/Earth-Intelligence-Lab/LocalizedWeather)
- Paper: Yang, Q., et al. (2024). *Local Off-Grid Weather Forecasting with Multi-Modal Earth Observation Data.* [arXiv:2410.12938](https://arxiv.org/abs/2410.12938)

## Hardware

LOAF uses open source hardware with no vendor lock-in:

- **Sensor**: DIY ultrasonic anemometer
- **Logger**: Raspberry Pi with RS-485/SDI-12 interface and 3D printed enclosure for predictions
- **Power**: Solar panel + battery for remote deployment

## Use Cases

- Off-grid environmental monitoring sites
- Research locations without dedicated weather infrastructure
- Applications requiring forecast transparency and hardware specifications for reproducibility

## Features

- Corrects systematic biases in large-scale forecast models for local conditions
- Hardware-transparent infrastructure for reproducible research
- Clear documentation for researchers without ML engineering backgrounds
- Combines numerical weather predictions with station measurements

## Related Projects

- [offgrid-weather-station](https://github.com/vinthewrench/offgrid-weather-station) - Off-grid weather station project
- [QingStation](https://github.com/majianjia/QingStation) - Open source ultrasonic anemometer
- [DL1GLH Ultrasonic Anemometer](https://www.dl1glh.de/ultrasonic-anemometer.html) - DIY ultrasonic wind sensor design

## Documentation

See the full documentation here: https://pandoroml.github.io/LOAF/

### Development Plans

- [ML Pipeline Development Plan](plan/dev-plan-ml-pipeline.md) - Detailed plan for reproducing LocalizedWeather for Seattle/PNW

### How to Run

The pipeline is: **download data → train a model → serve forecasts over REST**. Commands
below assume you're in the repo root and have installed the package (`pip install -e ".[dev]"`).
They use `config/arlington.yaml` (DCA, IAD, BWI, HEF, MRB stations); swap in
`config/seattle.yaml` for the PNW region instead.

#### Quickest path: one command

`loaf-pipeline` chains download → train → serve for you:

```bash
loaf-pipeline --config software/config/arlington.yaml \
    --start-date 2024-10-01 --end-date 2024-12-31 --year 2024 --port 5000
```

It blocks on the running server at the end, so leave it running and query it from another
terminal (see step 4 below). Useful flags for resuming a partial run:

- `--skip-download` — reuse whatever's already in `data/`
- `--skip-train --checkpoint <path>` — skip straight to serving an existing checkpoint
- `--no-serve` — stop after training, print the checkpoint path, and exit
- `--use-hrrr` / `--use-era5` — also download and fuse gridded forecasts (see step 1)

Run `loaf-pipeline --help` for the full flag list (it's the union of the three commands below).

Run loaf-report-summary --runs-dir runs after any batch of training runs to summarize results all modeling attempts in /runs.

#### Or, step by step

Useful if you want to inspect data between steps, retrain from already-downloaded data, or
swap in a different checkpoint without retraining.

**1. Download station observations** (IEM/ASOS — no registration required):

```bash
loaf-download-iem --config software/config/arlington.yaml \
    --start-date 2024-10-01 --end-date 2024-12-31
```

This is the minimum needed to train. Optionally fuse gridded forecasts too — HRRR needs no
auth (AWS S3), ERA5 needs a free [CDS API key](https://cds.climate.copernicus.eu/) in
`~/.cdsapirc`:

```bash
loaf-download-hrrr --config software/config/arlington.yaml \
    --start-date 2024-10-01 --end-date 2024-12-31
loaf-download-era5 --config software/config/arlington.yaml \
    --start-year 2024 --end-year 2024
```

All three write into `data/` in the current directory by default.

**2. Train a model** (MPNN or ViT backbone):

```bash
loaf-train --config software/config/arlington.yaml --year 2024
# add --model-type vit, --epochs N, --use-hrrr, or --use-era5 as needed
```

Writes `best.pt` / `last.pt` checkpoints and a `train_log.csv` to `runs/arlington_<timestamp>/`.

**3. Serve forecasts over REST:**

```bash
loaf-serve --checkpoint runs/arlington_<timestamp>/best.pt --port 5000
```

**4. Query it** (e.g. Reagan National Airport, Arlington VA):

```bash
curl "http://localhost:5000/api/forecast?lat=38.8512&lon=-77.0402"
curl http://localhost:5000/api/forecast/all   # every station in the graph
curl http://localhost:5000/health
```

Each response includes multi-horizon `u`/`v` wind predictions plus derived `wind_speed` and
`wind_direction`, ready to wire into a Home Assistant REST sensor.

## License

MIT

## Citation

If you find this project useful for your research or applications, please kindly cite using this BibTeX::
```bibtex
@software{Johnson_LOAF_Local_Observations_2026,
author = {Johnson, Keenan and Kim, Susie},
month = jan,
title = {{LOAF (Local Observations and Atmospheric Forecasting)}},
url = {https://github.com/PandoroML/LOAF},
year = {2026}
}
```

## Contact

- Website: https://pandoro.today
- Email: pandoro@breadboardfoundry.com

---

Built by [Bread Board Foundry](https://breadboardfoundry.com)

## Cool Picture Archive

The first pcb went to fab for our ultrasonic anemometer build inspired by QingStation. The first version is very similiar to the Ultrasonic Anemometer in QingStation, but it simplified by removing parts we don't need for now and updated to included parts that are easily purchaseable, as some parts in Qing have gone end of life.

<img width="1862" height="1862" alt="a6b5123db6724e5dabc65f7b7b04693a_T" src="https://github.com/user-attachments/assets/96b25ef9-431f-46aa-9dcc-e8fb26e51bd8" />
