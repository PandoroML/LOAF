---
date: 2026-01-13
authors:
  - keenanjohnson
---

# Wind Sensors!

We're kicking off LOAF hardware development with wind sensors. Here's why wind comes first, why ultrasonic anemometers are the right approach, and what open source options exist today.

<!-- more -->

## Why Wind First?

The [MIT Earth Intelligence Lab paper](https://arxiv.org/abs/2410.12938) that inspired LOAF tested their multi-modal transformer approach on several weather variables. Their key finding: **combining gridded forecasts with local station observations reduces prediction error by up to 80%** compared to using gridded data alone.

Wind benefits the most from this approach. The paper tested on MADIS weather station data, fusing HRRR forecasts with local observations for wind speed, temperature, and dewpoint. Wind prediction showed the largest improvements because it has the highest spatial variability—what the model learns from a station 10km away tells you less about wind than about temperature.

This makes sense: wind is highly local. A 3km forecast grid can't capture how terrain, buildings, and vegetation affect airflow at your specific site. Temperature and humidity vary more gradually across space, but wind can differ substantially between two points 100 meters apart.

Wind also has immediate practical value:

- **Wildfire risk assessment**: Wind speed and direction drive fire behavior
- **Renewable energy**: Small wind turbines and solar trackers need local wind data
- **Outdoor operations**: Agriculture, construction, drone flights all depend on wind conditions

Starting with the variable that benefits most from hyperlocal correction lets us validate the full pipeline before expanding to other measurements.

## Why Ultrasonic Anemometers?

Traditional cup anemometers spin in the wind. Simple, proven, but limited:

- **Moving parts wear out**: Bearings degrade, especially in dusty or salty environments
- **Inertia**: Cups take time to spin up and slow down, missing gusts and rapid changes
- **Separate vane needed**: Cup anemometers measure speed only; you need a separate wind vane for direction
- **Low-wind threshold**: Most cup sensors can't measure below 1-2 m/s

Ultrasonic anemometers measure wind by timing sound pulses between transducers. Wind speeds up sound traveling downwind and slows it traveling upwind. Comparing transit times in multiple directions gives speed and direction simultaneously.

```
       T1          T2
        \         /
         \       /
          \     /
           \   /
        ────▼─▼────  reflector plate
           /   \
          /     \
         /       \

    Each transducer sends a pulse down to the reflector
    and receives the echo. Wind shifts the sound path.

    4 transducers in a square pattern → 2D wind vector
    Compare transit times across all 4 to get speed + direction
```

The advantages:

- **No moving parts**: Nothing to wear out or calibrate mechanically
- **Better low-wind sensitivity**: Can measure down to 0.1 m/s
- **Faster response**: Captures gusts and turbulence at 10+ Hz
- **Simultaneous 2D or 3D measurement**: Speed and direction from the same sensor
- **Lower maintenance**: No bearings to replace


## Why Open Source Hardware?

Commercial weather stations lock you into ecosystems. A Davis anemometer needs a Davis logger. Each vendor has proprietary data formats, cloud services, and replacement parts.

Open source hardware means:

- **No vendor lock-in**: Standard interfaces (RS-485, SDI-12, I2C) work with any logger
- **Repairable**: Full schematics, BOMs, and 3D print files let you fix or modify anything
- **Transparent**: You know exactly what's being measured and how
- **Reproducible**: Other researchers can build identical sensors for comparative studies
- **Evolvable**: The community can improve designs over time

For a project like LOAF that aims to democratize hyperlocal forecasting, proprietary hardware defeats the purpose.

## Survey of Open Source Ultrasonic Anemometers

Two projects stand out as serious DIY ultrasonic anemometer efforts:

### QingStation

[QingStation](https://github.com/majianjia/QingStation) by Jianjia Ma is a compact weather station originally designed for autonomous maritime drones. Key specs:

- 48mm circular PCB with STM32L476 microcontroller
- 2x2 ultrasonic transducer array (40kHz/200kHz)
- Full sensor suite: barometer, humidity, temperature, compass, light, lightning detection
- Low power: ~20mA typical operation
- Well-documented with open source firmware

**Limitations for LOAF:**

- **No battery power option**: Designed for solar-powered boats with continuous charging. No documented battery operation or sleep modes for intermittent deployment.
- **Too integrated**: The all-in-one design bundles compute, sensors, and communication on a single PCB. For LOAF, we want modular sensors that connect to a separate Raspberry Pi logger. This allows mixing sensor types, easier repairs, and running ML inference on the Pi.
- **Parts sourcing issues**: Some components (specific transducers, lightning sensor) are difficult to source currently. Supply chain constraints make exact replication challenging.

### DL1GLH Ultrasonic Anemometer

[DL1GLH's project](https://www.dl1glh.de/ultrasonic-anemometer.html) by a German radio amateur represents over a decade of refinement:

- Achieves ±0.3 m/s accuracy (±2% RMS) and ±2° direction accuracy
- Uses pulse compression and Kalman filtering for robust signal processing
- Eight billion logged measurements validating 0.9998 correlation with reference instruments
- Extensive documentation of theory, calibration, and field testing

**Limitations:**

Minaly, although interesting, the DL1GLH project by Hardy isn not exactly easy to fabricate.

The documentation is clear and the theory is explained well, but some of the fab files are missing 
and some of the non core tech choices don't match the modern open-source standards.

### Other Projects

- **[ESP32 Ultrasonic Anemometer](https://hackaday.io/project/176848-ultrasonic-wind-sensor)**: Simpler ESP32-based design, less validated accuracy
- **[Hardy's Ultrasonic Anemometer](https://soldernerd.com/arduino-ultrasonic-anemometer/)**: Arduino-based, good learning project but not production-ready

## Our Approach

Given the limitations of existing projects, LOAF will develop a modular ultrasonic anemometer design:

1. **Sensor module**: Standalone ultrasonic wind sensor with standard digital output (RS-485 or I2C)
2. **Raspberry Pi logger**: Handles data collection, HRRR downloads, and ML inference
3. **Currently available parts**: BOM uses only components readily available from major distributors
4. **Battery + solar**: Designed for remote deployment from day one

We're starting with QingStation's transducer approach and DL1GLH's signal processing insights, adapted for our modular architecture.

## What's Next

Next post will cover the initial hardware prototype: transducer selection, PCB design decisions, and first bench tests. Stay tuned.
