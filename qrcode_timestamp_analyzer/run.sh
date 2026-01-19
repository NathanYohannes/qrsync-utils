#!/bin/bash
# QR Code Timestamp Analyzer - Quick Run Script

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Default port
PORT="${PORT:-5000}"

# Parse arguments
VIDEO_DIR="${1:-.}"
SHIFT_DONE=false

if [ "$1" = "--port" ] && [ -n "$2" ]; then
    PORT="$2"
    VIDEO_DIR="${3:-.}"
elif [ -n "$1" ]; then
    VIDEO_DIR="$1"
fi

echo "Starting QR Code Timestamp Analyzer..."
echo "  Video directory: $VIDEO_DIR"
echo "  Port: $PORT"
echo "  URL: http://127.0.0.1:$PORT"
echo ""

python frame_compare_web.py --dir "$VIDEO_DIR" --port "$PORT"
