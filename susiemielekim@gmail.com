# GRAIN (Gridded Regional Atmospheric Intelligence Network)

## What GRAIN Does

GRAIN generates hyperlocal weather forecasts for locations without nearby weather stations by combining:
- Regional gridded forecast models (e.g. NOAA GFS, HRRR)
- Sparse local weather station observations
- Multi-modal transformer architecture for spatial-temporal fusion

*Use cases:*
- Off-grid environmental monitoring (kelp farms, wildlife habitats, remote sensors)
- Research sites without dedicated weather infrastructure
- Applications requiring forecast transparency and hardware specifications for reproducibility

*Built on research from:*
Earth Intelligence Lab (October 2024) - Multi-modal transformers for hyperlocal weather prediction with sparse station networks.

## Key Features

- Hardware transparency: Complete specifications for reproducible research
- Domain accessibility: Documentation for domain scientists, not just ML engineers
- Clean energy: Development infrastructure powered by Washington state hydroelectric grid
- Open source: Permissive licensing for academic research, commercial licensing available

Part of [Pandoro](https://pandoro.today), Bread Board Foundry's research initiative building accessible ML infrastructure for climate and environmental research.
