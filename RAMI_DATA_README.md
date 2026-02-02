# RAMI Validation Data

This directory contains validation data from DART (Discrete Anisotropic Radiative Transfer) model simulations used to benchmark the LiDAR waveform simulator.

## Scene Description

The RAMI scene represents a 100m × 100m forest plot with 18 trees. The scene is divided into a 3×3 grid of footprints, each representing a different spatial location.

## Footprint Layout

```
┌─────────┬─────────┬─────────┐
│ RAMI_lu │ RAMI_up │ RAMI_ru │  Upper row
├─────────┼─────────┼─────────┤
│RAMI_Left│RAMI_mid │RAMI_right Middle row
├─────────┼─────────┼─────────┤
│ RAMI_ld │RAMI_down│ RAMI_rd │  Lower row
└─────────┴─────────┴─────────┘
```

**Footprint Centers (x, y coordinates in meters):**
- RAMI_lu: (22.5, 82.5)
- RAMI_up: (52.5, 82.5)
- RAMI_ru: (82.5, 82.5)
- RAMI_Left: (22.5, 52.5)
- RAMI_mid: (52.5, 52.5) - Center
- RAMI_right: (82.5, 52.5)
- RAMI_ld: (22.5, 22.5)
- RAMI_down: (52.5, 22.5)
- RAMI_rd: (82.5, 22.5)

## Data Files in Each Folder

Each RAMI_* folder contains:

### Main Waveform Files

1. **LIDAR_CONVOLVED_wave.txt**
   - Full waveform after pulse convolution
   - Includes multiple scattering effects
   - Format: Height (m) | Intensity

2. **LIDAR_CONVOLVED_wave_1stOrder.txt**
   - First-order scattering only
   - Format: Height (m) | Intensity

3. **LIDAR_DART_wave.txt**
   - Raw DART simulation output (all orders)
   - Before pulse convolution
   - Format: Height (m) | Intensity

4. **LIDAR_DART_wave_1stOrder.txt**
   - Raw first-order returns
   - Before pulse convolution
   - Format: Height (m) | Intensity

5. **pulse.txt**
   - Laser pulse shape used for convolution
   - Gaussian pulse with FWHM ≈ 0.57 ns
   - Format: Time (ns) | Amplitude

6. **stat_illumination_MC0.txt**
   - Monte Carlo illumination statistics
   - DART simulation metadata

### Additional Data

7. **LIDAR/IMAGES_DART/**
   - LiDAR imagery files from DART
   - Binary format (.gr#, .grf, .mp#, .mpr)
   - Visualization data

## Tree Parameters

The 18 trees in the scene have varying characteristics:

- **Heights**: 5.91 - 30.51 m
- **Crown bases**: 1.82 - 18.37 m
- **Crown radii**: 1.60 - 3.60 m
- **Leaf area**: Variable per tree (see rami_tree_data.py)

Detailed tree parameters are stored in [rami_tree_data.py](../rami_tree_data.py).

## Usage

The validation data is automatically loaded by the simulator when running RAMI examples:

```python
from lidar_waveform_simulator import example_rami_validation

# Run validation for center footprint
example_rami_validation('RAMI_mid')
```

The simulator compares its output against the DART reference waveforms to validate accuracy.

## DART Model Information

**DART (Discrete Anisotropic Radiative Transfer)** is a 3D radiative transfer model that simulates:
- Light propagation in atmosphere and vegetation
- Multiple scattering effects
- LiDAR waveform generation
- High accuracy reference data

**Reference:**
Gastellu-Etchegorry, J. P., et al. (2015). "Discrete Anisotropic Radiative Transfer (DART 5) for modeling airborne and satellite spectroradiometer and LIDAR acquisitions of natural and urban landscapes." Remote Sensing, 7(2), 1667-1701.

## Data Format Notes

- All height values are in meters above ground level
- Intensity values are normalized
- Text files use space or tab delimiters
- Coordinate system: Origin at (0,0) in lower-left corner
- Z-axis points upward (positive heights)

## Citation

If you use this validation data, please cite both:
1. This simulator repository
2. The DART model (see reference above)
