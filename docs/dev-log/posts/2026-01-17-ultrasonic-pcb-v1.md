---
date: 2026-01-17
authors:
  - keenanjohnson
---

# Submitted First Ultrasonic Anemometer PCB

Today I submitted the first version of the ultrasonic anemometer PCB for fabrication. This is a big milestone for the LOAF project.

<!-- more -->

## Design Origins

The PCB design is heavily inspired by the [QingStation project](https://github.com/majianjia/QingStation/tree/main), which is an open source ultrasonic weather station. Rather than starting from scratch, building on proven work made sense.

## Simplifications from QingStation

For this first version, I made several simplifications to focus on the core wind sensing functionality:

- **Removed the rain sensor** - not needed for initial wind measurement testing
- **Removed the microphone** - simplifies the analog design
- **Removed the lighting sensor** - reduces complexity
- **Removed the RGB sensor** - keeping the focus on ultrasonic transducers

The goal was to get a working ultrasonic anemometer with minimal complexity. Additional sensors can be added in future revisions once the core wind measurement is validated.

## Next Steps

While waiting for the boards to arrive, I'll be working on the data system side of LOAF. The goal is to set up the infrastructure for training and merging local models for improved weather predictions. Having the software pipeline ready means we can start collecting and processing data as soon as the hardware is operational.

Once the boards arrive:

1. Assemble and bring up the PCB
2. Test the ultrasonic transducer driving circuit
3. Validate wind speed and direction measurements
4. Compare accuracy against reference anemometers

Excited to see how this first spin performs.
