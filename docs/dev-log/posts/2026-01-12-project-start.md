---
date: 2026-01-12
authors:
  - keenanjohnson
---

# Introducing LOAF: Hyperlocal Weather Forecasting You Can Build

LOAF (Local Observations and Atmospheric Forecasting) aims to bring hyperlocal weather prediction to anyone with a Raspberry Pi and some sensors. Today I'm sharing why this project exists and where it's headed.

<!-- more -->

## The Problem

Weather forecasts work on grids. NOAA's HRRR model runs at 3km resolution. That's fine for knowing if Seattle will get rain, but useless for predicting wind conditions at a specific field site, a backyard wind turbine, or a wildfire evacuation route.

Research from MIT's Earth Intelligence Lab ([Yang et al., 2024](https://arxiv.org/abs/2410.12938)) demonstrated that combining gridded forecasts with sparse local station observations through a multi-modal transformer architecture can reduce prediction error by up to 80% compared to gridded data alone. Their approach fuses regional forecast models (HRRR, ERA5) with local weather station data to generate genuinely local predictions.

Commercial companies like Tomorrow.io and ClimaCell offer hyperlocal forecasts, but their systems are proprietary and require ongoing subscriptions. The MIT research code exists on GitHub, but there's no easy way to deploy your own sensor, train a model for your region, and run forecasts locally.

LOAF fills that gap: an open-source, end-to-end system from sensor to forecast.

## Why Start with Wind

Wind is the ideal first target:

- **Validated approach**: The MIT paper specifically tested on wind data from MADIS weather stations
- **High value**: Wind prediction matters for wildfire danger assessment, renewable energy, and outdoor operations
- **Simpler validation**: Ground truth is easier to verify than precipitation or cloud cover
- **Existing data**: HRRR provides 3km resolution wind forecasts we can downscale

We'll forecast near-surface wind speed (m/s) and direction (degrees), 6-48 hours ahead with hourly updates.

## Beyond Wind

Once the wind pipeline works, the architecture generalizes. Temperature, humidity, and precipitation follow the same pattern: gridded forecast + local observations → transformer fusion → hyperlocal prediction. The goal is a complete local weather station that improves on regional forecasts for all variables.

## Why Open Source Hardware

Commercial weather stations lock you into proprietary data loggers and cloud services. A Davis anemometer needs a Davis logger. HOBO sensors need HOBO infrastructure.

LOAF uses commodity hardware:

- Raspberry Pi for data logging and inference
- Standard sensors with documented protocols (RS-485, SDI-12)
- Solar power for remote deployment
- Complete bill of materials with part numbers

Anyone can build, modify, and repair the system. No vendor lock-in, no subscription fees, no cloud dependency.

## Wind Sensor Research: Going Ultrasonic

Traditional cup anemometers have moving parts that wear out, require regular calibration, and struggle with low wind speeds. Ultrasonic anemometers measure wind by timing sound pulses between transducers—no moving parts, better low-wind sensitivity, and simultaneous speed/direction measurement.

Commercial ultrasonic sensors cost $600-1600. But two open-source projects prove DIY ultrasonic anemometers are viable:

**[QingStation](https://github.com/majianjia/QingStation)** by Jianjia Ma is a compact weather station designed for autonomous maritime drones. It integrates a 2×2 ultrasonic transducer array (40kHz/200kHz) with reflection-based measurement on a 48mm circular PCB drawing ~20mA. The design proves ultrasonic wind measurement works in a small, low-power package.

**[DL1GLH's Ultrasonic Anemometer](https://www.dl1glh.de/ultrasonic-anemometer.html)** is a decade-long project achieving ±0.3 m/s accuracy (±2% RMS) for speed and ±2° for direction. The second prototype uses pulse compression, Kalman filtering, and sound reflection to handle turbulence. Eight billion measurements logged during field testing showed 0.9998 correlation against reference instruments.

These projects demonstrate that with careful signal processing, a 3D-printed housing and ~$50 in ultrasonic transducers can match commercial sensor accuracy.

## What's Next

1. **Hardware assembly**: Raspberry Pi + ultrasonic anemometer prototype
2. **Data pipeline**: Automated HRRR download, MADIS station integration
3. **Model training**: Adapt MIT's transformer architecture for Seattle area
4. **Edge deployment**: Run inference on the Pi itself

First sensor goes up this month. Updates to follow.
