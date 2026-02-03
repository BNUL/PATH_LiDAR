@echo off
REM PATH LiDAR Waveform Simulator v2.0 - Windows Release Test Script

echo ============================================================
echo   PATH LiDAR Waveform Simulator v2.0
echo   Release Date: 2026-02-03
echo ============================================================
echo.

REM Test Python availability
python --version >nul 2>&1
if errorlevel 1 (
    echo X Python not found. Please install Python 3.7+
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYVER=%%i
echo + Python found: %PYVER%

REM Test dependencies
echo.
echo Testing dependencies...
python -c "import numpy; import scipy; import matplotlib; print('+ NumPy, SciPy, Matplotlib installed')" || exit /b 1

echo.
echo Testing GDAL/rasterio (for CHM support)...
python -c "try: from osgeo import gdal; print('+ GDAL installed')^
except: import rasterio; print('+ rasterio installed')" 2>nul || (
    echo ! Warning: Neither GDAL nor rasterio found. CHM features will not work.
    echo    Install with: pip install gdal  OR  pip install rasterio
)

echo.
echo Testing core modules...
python -c "from lidar_simulator_core import CHMWaveformSimulator, LiDARWaveformSimulator; from chm_waveform_simulator import simulate_from_chm; print('+ Core modules imported successfully')" || exit /b 1

echo.
echo Checking files...
for %%f in (lidar_simulator_core.py lidar_waveform_simulator.py chm_waveform_simulator.py 6_chm.tif README.md) do (
    if exist "%%f" (
        echo + %%f
    ) else (
        echo X Missing: %%f
        exit /b 1
    )
)

echo.
echo ============================================================
echo + All checks passed! Ready to use.
echo ============================================================
echo.
echo Quick start:
echo   python lidar_waveform_simulator.py  # Run all examples
echo   python chm_waveform_simulator.py    # CHM demo
echo.
echo See VERSION_2.0_README.md for full documentation.
echo.
pause
