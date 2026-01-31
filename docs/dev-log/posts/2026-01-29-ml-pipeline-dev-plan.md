---
date: 2026-01-29
authors:
  - keenanjohnson
---

# ML Pipeline Development Plan

Today I finalized the development plan for the LOAF ML pipeline. This plan outlines how we'll reproduce the LocalizedWeather paper for the Seattle/Pacific Northwest region with continuous forecasting capabilities.

<!-- more -->

## Key Decisions

### Phased Approach

We're taking a two-phase approach:

1. **Phase 1: MADIS Data** - First, we'll build the full ML pipeline using existing MADIS weather station observations. This lets us validate the model architecture and training pipeline using publicly available ground truth data from ~50-100 stations in the PNW region.

2. **Phase 2: Raspberry Pi Integration** - Once the pipeline is working with MADIS data, we'll integrate our own Raspberry Pi sensor hub. This adds our custom ultrasonic anemometer observations to improve hyperlocal predictions at our specific deployment site.

### Two-Computer Architecture

We're separating concerns between a Linux server (for ML training and inference) and a Raspberry Pi (for sensor data collection). This keeps the heavy ML workload off the edge device while still enabling local sensor integration.

```
Linux Server (ML)          Raspberry Pi (Sensors) [Phase 2]
├── Data download          ├── Ultrasonic anemometer
├── Model training         ├── Data logging
├── Hourly inference       └── Upload to server
└── Home Assistant API
```

### Data Sources

We're using three primary data sources matching the original paper:

- **ERA5**: 31 km global reanalysis baseline (CDS API)
- **HRRR**: 3 km high-resolution regional forecasts (AWS S3)
- **MADIS**: Ground truth station observations (NOAA)

Seattle region bounds: 46.5°N to 49.0°N, -124.0°W to -121.0°W

### Model Architecture

The full GNN + ViT architecture from the LocalizedWeather paper:

1. **Heterogeneous MPNN** - Message passing between stations and grid points
2. **Vision Transformer** - Self-attention over spatial tokens
3. **Output head** - Wind predictions at multiple lead times (6h to 48h)

## Implementation Milestones

**Phase 1: MADIS-based pipeline**
1. **Data Pipeline** - Package structure, data downloaders, loaders
2. **Model Architecture** - Port GNN and transformer components
3. **Training Pipeline** - Dataset class, training loop, evaluation
4. **Inference & API** - REST server for forecasts
5. **Home Assistant Integration** - Weather entity or REST sensor
6. **Continuous Operation** - Systemd services, cron scheduling

**Phase 2: Custom sensors**
7. **Pi Sensor Hub** - Raspberry Pi with ultrasonic anemometer integration

## Next Steps

Before we start coding:

1. Register for ERA5/CDS access
2. Register for MADIS data access
3. Verify GPU availability and CUDA version
4. Ensure ~500GB free disk space
5. Clone LocalizedWeather repo for reference

The first coding task will be setting up the package structure and implementing the HRRR download functionality.

## Resources

- [Full Development Plan](../../../plan/dev-plan-ml-pipeline.md)
- [LocalizedWeather Paper](https://arxiv.org/abs/2410.12938)
- [LocalizedWeather GitHub](https://github.com/Earth-Intelligence-Lab/LocalizedWeather)
