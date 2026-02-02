# LiDAR Waveform Simulator

A Python-based full-waveform LiDAR simulator for vegetation canopy analysis using the PATH (Photon tracing for Atmosphere and canopy Height) model. This simulator generates realistic LiDAR waveforms for nadir-looking instruments over forest canopies.

## Features

- **Multiple Crown Shapes**: Support for cylinder, sphere, and cone crown geometries
- **Flexible Canopy Configuration**: Customizable tree parameters including crown density, size, height, and leaf properties
- **Physics-Based Simulation**: Implements gap probability theory and multiple scattering effects
- **Pulse Convolution**: Realistic waveform generation with custom pulse shapes
- **Validation Dataset**: Includes RAMI (RAdiation transfer Model Intercomparison) benchmark data
- **Sensitivity Analysis**: Built-in tools for parameter sensitivity analysis

## Installation

### Prerequisites

- Python 3.7 or higher
- NumPy
- SciPy
- Matplotlib

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/lidar-waveform-simulator.git
cd lidar-waveform-simulator
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── lidar_simulator_core.py       # Core simulation engine and classes
├── lidar_waveform_simulator.py   # Main simulator with examples
├── rami_tree_data.py              # RAMI scene tree parameters
├── sensitivity_analysis.py        # Parameter sensitivity analysis tools
├── RAMI_*/                        # RAMI benchmark data folders
│   ├── LIDAR_CONVOLVED_wave.txt
│   ├── LIDAR_DART_wave.txt
│   ├── pulse.txt
│   └── LIDAR/IMAGES_DART/
└── requirements.txt
```

## Quick Start

### Basic Usage

```python
from lidar_simulator_core import (
    CanopyParameters,
    SimulationConfig,
    LiDARWaveformSimulator
)

# Define canopy parameters
canopy = CanopyParameters(
    crown_shape='cylinder',
    crown_density=0.02,      # trees/m²
    crown_radius=3.0,        # meters
    crown_height_range=(11.0, 17.0),  # (base, top) in meters
    leaf_area_density=0.6,   # m²/m³
    leaf_reflectance=0.5
)

# Configure simulation
config = SimulationConfig(
    max_height=40.0,
    height_resolution=0.15,
    fwhm=0.57,
    enable_multiple_scattering=True
)

# Run simulation
simulator = LiDARWaveformSimulator(canopy, config)
results = simulator.run()

# Visualize results
simulator.plot_results()
```

### Running Examples

The package includes several example scenarios:

```python
python lidar_waveform_simulator.py
```

This will run demonstrations for:
- Cylinder crown shape
- Sphere crown shape  
- Cone crown shape
- RAMI validation scenes

### Sensitivity Analysis

Perform parameter sensitivity analysis:

```python
python sensitivity_analysis.py
```

This generates figures showing the impact of various parameters (crown density, leaf area density, crown radius, etc.) on the simulated waveforms.

## RAMI Validation Data

The `RAMI_*` folders contain benchmark data from DART (Discrete Anisotropic Radiative Transfer) model simulations for validation:

- **RAMI_lu, RAMI_up, RAMI_ru**: Upper row footprints (left, center, right)
- **RAMI_Left, RAMI_mid, RAMI_right**: Middle row footprints
- **RAMI_ld, RAMI_down, RAMI_rd**: Lower row footprints (left, center, right)

Each folder contains:
- `LIDAR_CONVOLVED_wave.txt`: Convolved waveform data
- `LIDAR_DART_wave.txt`: DART raw waveform
- `pulse.txt`: Pulse shape data
- `LIDAR/IMAGES_DART/`: Additional imagery data

## Key Components

### CanopyParameters

Defines the vegetation canopy structure:

- `crown_shape`: 'cylinder', 'sphere', or 'cone'
- `crown_density`: Tree density (trees/m²)
- `crown_radius`: Crown horizontal extent (m)
- `crown_height_range`: (base_height, top_height) tuple (m)
- `leaf_area_density`: Leaf area per crown volume (m²/m³)
- `leaf_angle_distribution`: LAD type (6 = spherical)
- `leaf_reflectance`: Leaf reflectance coefficient
- `ground_reflectance`: Ground surface reflectance

### SimulationConfig

Controls simulation parameters:

- `max_height`: Maximum height of simulation domain (m)
- `height_resolution`: Vertical discretization (m)
- `fwhm`: Full width at half maximum of pulse (ns)
- `enable_multiple_scattering`: Include multiple scattering effects
- `num_monte_carlo`: Number of Monte Carlo samples

### LiDARWaveformSimulator

Main simulation class that:
1. Calculates gap probability profiles
2. Computes energy attenuation through canopy
3. Simulates ground and canopy returns
4. Convolves with instrument pulse shape
5. Validates against reference data (if available)

## Theory

The simulator implements:

1. **Gap Probability Model**: Uses exponential attenuation based on path-length distribution within crowns
2. **Turbid Medium Approximation**: Treats canopy as a continuous medium with specified leaf area density
3. **Multiple Scattering**: Optional second-order scattering effects
4. **G-function**: Accounts for leaf angle distribution effects on interception
5. **Pulse Convolution**: Realistic pulse broadening using Gaussian pulses

## Citation

If you use this simulator in your research, please cite:

```bibtex
@software{lidar_waveform_simulator,
  title={LiDAR Waveform Simulator for Vegetation Canopy Analysis},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/lidar-waveform-simulator}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- RAMI validation data derived from DART model simulations
- PATH model theoretical framework
- GLAS/ICESat mission for pulse shape parameters

## Contact

For questions, issues, or contributions, please open an issue on GitHub or contact [your.email@institution.edu].

## References

1. Ni-Meister, W., et al. (2001). A clumped-foliage canopy radiative transfer model for a Global Dynamic Terrestrial Ecosystem Model
2. DART model: Gastellu-Etchegorry, J. P., et al. (2015). Discrete Anisotropic Radiative Transfer (DART 5) for modeling airborne and satellite spectroradiometer and LIDAR acquisitions
3. GLAS/ICESat: Zwally, H. J., et al. (2002). ICESat's laser measurements of polar ice, atmosphere, ocean, and land
