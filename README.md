# LOAF (Local Observations and Atmospheric Forecasting)

Open source hyperlocal weather forecasting combining machine learning forecast models with local station observations.

- Do you want to improve the weather forecasts at a specific place? 
- Do you have access to local sensors or are intersted in building them? 
- Are you interested in understanding the full process and not relying on big tech companies hiding proprietary algorithms to do so? 

If you answered yes to all of the above, then LOAF is for you.

Part of [Pandoro](https://pandoro.today) — Bread Board Foundry's open science ML tools for climate and environmental research.

This is definitely a work in progress, so stay tuned!

## Latest Cool Picture

The first pcb went to fab for our ultrasonic anemometer build inspired by QingStation. The first version is very similiar to the Ultrasonic Anemometer in QingStation, but it simplified by removing parts we don't need for now and updated to included parts that are easily purchaseable, as some parts in Qing have gone end of life.

<img width="1862" height="1862" alt="a6b5123db6724e5dabc65f7b7b04693a_T" src="https://github.com/user-attachments/assets/96b25ef9-431f-46aa-9dcc-e8fb26e51bd8" />


## Current priorities:

- Build DIY ultrasonic anemometer (inspired by [QingStation](https://github.com/majianjia/QingStation) and [DL1GLH](https://www.dl1glh.de/ultrasonic-anemometer.html))
- Deploy initial wind sensor in Seattle area
- Set up HRRR/MADIS data pipeline
- Train regional model and validate against local observations
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
