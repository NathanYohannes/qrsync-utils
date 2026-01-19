#!/bin/bash
# QR Calibration Tools - Combined Setup Script
# Installs the QR beacon, camera setup tool, and timestamp analyzer

set -e

echo "========================================"
echo "  QR Calibration Tools Setup"
echo "========================================"
echo ""

# Check Python version
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found. Please install Python 3.8 or later."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Found Python $PYTHON_VERSION"
echo ""

# Setup display_qrcode
echo "[1/3] Setting up QR Timestamp Beacon (display_qrcode)..."
cd display_qrcode
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
deactivate
cd ..
echo "      ✓ QR Timestamp Beacon ready"
echo ""

# Setup sync_camera_setup
echo "[2/3] Setting up Sync Camera Setup..."
cd sync_camera_setup
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
deactivate
cd ..
echo "      ✓ Sync Camera Setup ready"
echo ""

# Setup qrcode_timestamp_analyzer
echo "[3/3] Setting up Timestamp Analyzer..."
cd qrcode_timestamp_analyzer
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Optional zxing-cpp
echo ""
echo "Would you like to install zxing-cpp for better QR detection? (y/N)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Installing zxing-cpp..."
    pip install zxing-cpp
    # Also install in sync_camera_setup
    deactivate
    cd ../sync_camera_setup
    source venv/bin/activate
    pip install zxing-cpp
fi
deactivate
cd ..
echo "      ✓ Timestamp Analyzer ready"

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "USAGE:"
echo ""
echo "1. QR Timestamp Beacon (displays QR codes with timestamps):"
echo "   ./run_beacon.sh        # Starts on port 8080"
echo ""
echo "2. Sync Camera Setup (live camera preview + QR detection):"
echo "   ./run_camera_setup.sh  # Starts on port 5001"
echo ""
echo "3. Timestamp Analyzer (analyze recorded videos):"
echo "   ./run_analyzer.sh /path/to/videos  # Starts on port 5000"
echo ""
echo "WORKFLOW:"
echo "  1. Run beacon on display device"
echo "  2. Run camera setup to verify QR detection"
echo "  3. Record videos with cameras"
echo "  4. Run analyzer to measure sync drift"
echo ""
