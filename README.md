# LOAF (Local Observations and Atmospheric Forecasting)

Open source hyperlocal weather forecasting combining gridded forecasts with local station observations using multi-modal transformers.

Part of [Pandoro](https://pandoro.today) — Bread Board Foundry's open science ML tools for climate and environmental research.

---

## About

LOAF generates hyperlocal weather forecasts for locations without nearby weather stations by combining:

- Regional gridded forecast models (NOAA GFS, HRRR, ERA5)
- Sparse local weather station observations
- Multi-modal transformer architecture for spatial-temporal fusion

**Built on research from MIT Earth Intelligence Lab:**
- GitHub: [Earth-Intelligence-Lab/LocalizedWeather](https://github.com/Earth-Intelligence-Lab/LocalizedWeather)
- Paper: Yang, Q., et al. (2024). *Local Off-Grid Weather Forecasting with Multi-Modal Earth Observation Data.* [arXiv:2410.12938](https://arxiv.org/abs/2410.12938)

LOAF provides a production-quality implementation with hardware transparency and documentation for domain scientists.

## Use Cases

- Off-grid environmental monitoring sites
- Research locations without dedicated weather infrastructure
- Applications requiring forecast transparency and hardware specifications for reproducibility

## Features

- Corrects systematic biases in large-scale forecast models for local conditions
- Hardware-transparent infrastructure for reproducible research
- Clear documentation for researchers without ML engineering backgrounds
- Combines numerical weather predictions with station measurements

## Getting Started

[Installation and usage documentation coming soon]

## Documentation

[Link to full documentation coming soon]

## Contributing

[Contribution guidelines coming soon]

## License

[License information coming soon]

## Citation

If you use LOAF in your research, please cite both this implementation and the original research:
```bibtex
[Citation format coming soon]
```

## Contact

- Website: https://pandoro.today
- Email: pandoro@breadboardfoundry.com

---

Built by [Bread Board Foundry](https://breadboardfoundry.com)
