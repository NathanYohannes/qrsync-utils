#!/bin/bash
# QR Calibration Tools - Combined Setup Script
# This installs both the QR beacon display and the timestamp analyzer

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
echo "[1/2] Setting up QR Timestamp Beacon (display_qrcode)..."
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

# Setup qrcode_timestamp_analyzer
echo "[2/2] Setting up Timestamp Analyzer..."
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
echo "1. QR Timestamp Beacon (for displaying QR codes):"
echo "   cd display_qrcode"
echo "   source venv/bin/activate"
echo "   python main.py --port 8080"
echo "   → Open http://localhost:8080 in browser"
echo ""
echo "2. Timestamp Analyzer (for analyzing videos):"
echo "   cd qrcode_timestamp_analyzer"
echo "   source venv/bin/activate"
echo "   python frame_compare_web.py --dir /path/to/videos"
echo "   → Open http://localhost:5000 in browser"
echo ""
echo "Or use the quick-run scripts:"
echo "   ./run_beacon.sh        # Start QR beacon on port 8080"
echo "   ./run_analyzer.sh DIR  # Start analyzer with video directory"
echo ""
