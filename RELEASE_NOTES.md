# PATH LiDAR Waveform Simulator - Release Notes

## Version 2.0 - 2026-02-03

### 🎉 Major Features

#### 1. **CHM-Based Waveform Simulation** (New!)
- **NEW**: `CHMWaveformSimulator` class for simulating LiDAR waveforms from Canopy Height Model (CHM) raster data
- Automatically extracts path length distribution from CHM structure
- Supports manual gap probability (`manual_Fs0`) and tree top height adjustment (`manual_tree_top_height`)
- Compatible with GeoTIFF format CHM files
- Example CHM data included: `6_chm.tif` (23m forest scene)

#### 2. **Simplified API**
- Wrapper function `simulate_from_chm()` in `chm_waveform_simulator.py` for easy CHM-based simulation
- Integrated CHM example (`example_chm_based()`) into main demo file `lidar_waveform_simulator.py`

#### 3. **Core Improvements**
- Removed tree height distribution convolution (simplified wave calculation)
- Removed percentile-based filtering (now uses mean crown length directly)
- Fixed bin allocation logic to capture all crown lengths
- Dynamic `layers_v` adjustment based on actual max path length
- Cleaned up gap probability output (removed verbose model info for CHM mode)

### 📦 Package Structure

```
PATH_LiDAR/
├── lidar_simulator_core.py          # Core simulation engine (1857 lines)
│   ├── CanopyParameters              # Tree canopy configuration
│   ├── SimulationConfig              # Simulation settings
│   ├── LiDARWaveformSimulator        # Standard waveform simulator
│   └── CHMWaveformSimulator          # NEW: CHM-based simulator
│
├── lidar_waveform_simulator.py      # Demo examples (487 lines)
│   ├── example_cylinder()            # Cylindrical crown
│   ├── example_sphere()              # Spherical crown
│   ├── example_cone()                # Conical crown
│   ├── example_with_height_distribution()  # Real tree structure
│   ├── example_with_pulse()          # Gaussian pulse convolution
│   ├── example_chm_based()           # NEW: CHM-based simulation
│   ├── example_with_dart_pulse()     # DART pulse file
│   └── example_rami_scene()          # RAMI standard scenes
│
├── chm_waveform_simulator.py        # CHM wrapper (155 lines)
│   └── simulate_from_chm()           # High-level CHM simulation API
│
├── rami_tree_data.py                 # RAMI scene configurations
├── sensitivity_analysis.py           # Parameter sensitivity plots
│
├── 6_chm.tif                         # Example CHM data (NEW)
├── RAMI_*/                           # RAMI validation scenes (9 folders)
│
└── Documentation:
    ├── README.md                     # Main documentation
    ├── CONTRIBUTING.md               # Contribution guidelines
    └── RAMI_DATA_README.md           # RAMI data description
```

### 🔧 Technical Changes

#### CHM Simulation Algorithm:
1. **Load CHM**: Read GeoTIFF using GDAL/rasterio
2. **Calculate Parameters**:
   - `layers_vb = ceil(tree_top_height / layer_height)`
   - `layers_v = ceil(mean_crown_length / layer_height)` (dynamic)
   - Branch height = mean(CHM) × branch_height_ratio
   - Crown length = CHM - branch_height
3. **Path Length Distribution**:
   - Bins based on mean crown length (not percentile)
   - Last bin captures all values > (n_bins-1) × layer_height
   - Auto-adjust `n_bins` to ensure `n_bins >= layers_v + 1`
4. **Gap Probability**: Auto-calculated from CHM or manually set
5. **Waveform Generation**: Same physics as standard simulator + GEDI pulse convolution

#### Recommended Parameters for Strong Signal:
```python
leaf_area_density = 0.8        # (was 0.4, now 0.8-1.0 for 20m+ forests)
leaf_reflectance = 0.57        # (was 0.35, now NIR typical value)
ground_reflectance = 0.2       # (reduce to highlight vegetation)
```

### 📝 API Changes

#### New Function:
```python
from chm_waveform_simulator import simulate_from_chm

height, waveform = simulate_from_chm(
    chm_file="6_chm.tif",
    layer_height=0.15,              # 0.15m resolution
    height_threshold=1.0,           # Gap detection threshold
    branch_height_ratio=1/3,        # Lower 1/3 is trunk
    manual_Fs0=None,                # Auto-calculate gap probability
    manual_tree_top_height=None,    # Auto-use max(CHM)
    leaf_area_density=0.8,          # Recommended for dense forests
    leaf_reflectance=0.57,          # NIR reflectance
    use_pulse_shape=True,           # GEDI pulse convolution
    plot=True
)
```

#### New Class:
```python
from lidar_simulator_core import CHMWaveformSimulator, CanopyParameters

canopy = CanopyParameters(
    leaf_area_density=0.8,
    leaf_reflectance=0.57,
    ...
)

simulator = CHMWaveformSimulator(
    chm_file="6_chm.tif",
    canopy_params=canopy,
    layer_height=0.15,
    height_threshold=1.0,
    branch_height_ratio=1/3
)

height, waveform = simulator.simulate(plot=True)
```

### 🗑️ Removed Files (Cleanup)
- `CHM_WAVEFORM_README.md` (consolidated into README.md)
- `CHM_SUMMARY.md` (redundant)
- `TREE_GROWTH_GUIDE.md` (feature removed)
- `PERCENTILE_GUIDE.md` (feature removed)
- `chm_percentile_example.py` (no longer needed)
- `chm_tree_growth_example.py` (no longer needed)
- `chm_quickstart.py` (integrated into main examples)
- Development notes (ZENODO_GUIDE.md, GITHUB_GUIDE.md, etc.)

### 🐛 Bug Fixes
- Fixed bin allocation missing values > max_path_length
- Fixed `layers_v` not updating when path length changes
- Fixed display DPI issue (300 → 100 for screen, 300 for save)
- Removed `bbox_inches='tight'` from savefig to maintain consistency

### 📊 Validation
- Tested with 6_chm.tif (23.38m forest, 12178 pixels)
- LAI calculation: 2.9-3.7 depending on parameters
- Gap probability: ~0.065 (6.5% gaps)
- Waveform range: -3m to 26m (pulse extension included)

### 🚀 Usage Examples

**Basic CHM Simulation:**
```python
python lidar_waveform_simulator.py  # Run example_chm_based()
```

**Custom CHM Simulation:**
```python
python chm_waveform_simulator.py    # Simple wrapper demo
```

**Traditional Simulation:**
```python
python lidar_waveform_simulator.py  # Run example_cylinder(), etc.
```

### 📚 Documentation Updates
- Main README.md updated with CHM simulation guide
- Added CHM parameter recommendations
- Simplified example structure
- Removed redundant documentation

### ⚙️ Requirements
- Python 3.7+
- numpy
- scipy
- matplotlib
- GDAL or rasterio (for CHM loading)

### 🙏 Acknowledgments
Based on the PATH (Photon transport in plant canopy) model. CHM feature developed to support integration with remote sensing data products.

---

## Version 1.0 - 2026-02-02
- Initial Python port from MATLAB
- Standard geometric crown shapes (cylinder, sphere, cone)
- RAMI scene validation
- DART pulse convolution support
- Sensitivity analysis tools
