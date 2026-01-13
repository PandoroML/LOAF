# Welcome to LOAF ((Local Observations and Atmospheric Forecasting))

LOAF generates hyperlocal weather forecasts for locations without nearby weather stations by combining:

Regional forecast models (NOAA GFS, HRRR, ERA5)
Sparse local weather station observations
Multi-modal transformer architecture for spatial-temporal fusion
Built on research from MIT Earth Intelligence Lab:

GitHub: Earth-Intelligence-Lab/LocalizedWeather
Paper: Yang, Q., et al. (2024). Local Off-Grid Weather Forecasting with Multi-Modal Earth Observation Data. arXiv:2410.12938

## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
