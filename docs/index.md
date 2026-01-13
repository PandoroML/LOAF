# LOAF (Local Observations and Atmospheric Forecasting)

LOAF generates hyperlocal weather forecasts for locations without nearby weather stations by combining regional forecast models with local sensor observations using multi-modal transformers.

**Built on research from MIT Earth Intelligence Lab:**

- GitHub: [Earth-Intelligence-Lab/LocalizedWeather](https://github.com/Earth-Intelligence-Lab/LocalizedWeather)
- Paper: Yang, Q., et al. (2024). [Local Off-Grid Weather Forecasting with Multi-Modal Earth Observation Data](https://arxiv.org/abs/2410.12938)

## Architecture

```mermaid
flowchart TB
    subgraph Data Sources
        HRRR[NOAA HRRR<br/>3km gridded forecasts]
        MADIS[MADIS Stations<br/>Regional observations]
        LOCAL[Local Sensor<br/>Ultrasonic anemometer]
    end

    subgraph Data Pipeline
        FETCH[Data Fetcher<br/>Hourly HRRR download]
        ALIGN[Alignment<br/>Spatial/temporal sync]
    end

    subgraph Model
        SPATIAL[Spatial Encoder<br/>Grid transformer]
        STATION[Station Encoder<br/>Observation tokens]
        FUSION[Cross-Attention<br/>Fusion module]
        DECODER[Decoder<br/>Local prediction]
    end

    subgraph Output
        FORECAST[6-48hr Forecast<br/>Wind speed & direction]
        HA[Home Assistant<br/>Widget display]
    end

    HRRR --> FETCH
    FETCH --> ALIGN
    MADIS --> ALIGN
    LOCAL --> ALIGN

    ALIGN --> SPATIAL
    ALIGN --> STATION

    SPATIAL --> FUSION
    STATION --> FUSION
    FUSION --> DECODER

    DECODER --> FORECAST
    FORECAST --> HA
```

## Data Flow

1. **Gridded Forecasts**: NOAA HRRR provides 3km resolution wind forecasts, downloaded hourly
2. **Station Observations**: MADIS network stations provide ground truth from the surrounding region
3. **Local Sensor**: DIY ultrasonic anemometer captures hyperlocal conditions
4. **Fusion**: Multi-modal transformer combines coarse forecasts with sparse observations
5. **Prediction**: Model outputs corrected forecasts for the exact sensor location

## Hardware Stack

| Component | Description |
|-----------|-------------|
| Sensor | DIY ultrasonic anemometer (40kHz transducers) |
| Logger | Raspberry Pi 4 with RS-485 interface |
| Power | 20W solar panel + 12V battery |
| Enclosure | 3D printed weatherproof housing |

## Model Architecture

The transformer architecture processes two input streams:

- **Spatial encoder**: Treats HRRR grid points as tokens, applies self-attention to capture regional patterns
- **Station encoder**: Each weather station becomes a token with positional encoding based on lat/lon
- **Cross-attention fusion**: Target location queries both encoders to aggregate relevant information
- **Decoder**: Predicts wind speed/direction at forecast horizons (6-48 hours)

Training uses historical HRRR forecasts paired with MADIS observations (2020-2024), validated against held-out local sensor data.
