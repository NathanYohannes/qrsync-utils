#!/bin/bash
# QR Code Timestamp Analyzer - Setup Script
# Run this on a new device to set up the environment

set -e

echo "=== QR Code Timestamp Analyzer Setup ==="
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

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Optional: Install zxing-cpp for better QR detection
echo ""
echo "Would you like to install zxing-cpp for improved QR detection? (y/N)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Installing zxing-cpp..."
    pip install zxing-cpp
fi

# Create recordings directory if it doesn't exist
mkdir -p recordings

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the web viewer:"
echo "  source venv/bin/activate"
echo "  python frame_compare_web.py --dir /path/to/videos"
echo ""
echo "Or use the run script:"
echo "  ./run.sh /path/to/videos"
echo ""
