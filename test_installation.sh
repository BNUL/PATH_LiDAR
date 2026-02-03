#!/bin/bash
# PATH LiDAR Waveform Simulator v2.0 - Release Script

echo "============================================================"
echo "  PATH LiDAR Waveform Simulator v2.0"
echo "  Release Date: 2026-02-03"
echo "============================================================"
echo ""

# Test Python availability
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.7+"
    exit 1
fi

echo "✓ Python found: $(python --version)"

# Test dependencies
echo ""
echo "Testing dependencies..."
python -c "import numpy; import scipy; import matplotlib; print('✓ NumPy, SciPy, Matplotlib installed')" || exit 1

echo ""
echo "Testing GDAL/rasterio (for CHM support)..."
python -c "try:
    from osgeo import gdal
    print('✓ GDAL installed')
except:
    try:
        import rasterio
        print('✓ rasterio installed')
    except:
        print('⚠️  Warning: Neither GDAL nor rasterio found. CHM features will not work.')
        print('   Install with: pip install gdal  OR  pip install rasterio')" || true

echo ""
echo "Testing core modules..."
python -c "
from lidar_simulator_core import CHMWaveformSimulator, LiDARWaveformSimulator
from chm_waveform_simulator import simulate_from_chm
print('✓ Core modules imported successfully')
" || exit 1

echo ""
echo "Checking files..."
files=("lidar_simulator_core.py" "lidar_waveform_simulator.py" "chm_waveform_simulator.py" "6_chm.tif" "README.md")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "❌ Missing: $file"
        exit 1
    fi
done

echo ""
echo "============================================================"
echo "✓ All checks passed! Ready to use."
echo "============================================================"
echo ""
echo "Quick start:"
echo "  python lidar_waveform_simulator.py  # Run all examples"
echo "  python chm_waveform_simulator.py    # CHM demo"
echo ""
echo "See VERSION_2.0_README.md for full documentation."
echo ""
