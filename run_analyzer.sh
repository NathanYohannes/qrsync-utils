#!/bin/bash
# Quick-run script for Timestamp Analyzer

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/qrcode_timestamp_analyzer"

if [ ! -d "venv" ]; then
    echo "Error: Run ./setup.sh first"
    exit 1
fi

source venv/bin/activate

VIDEO_DIR="${1:-.}"
PORT="${2:-5000}"

echo "📊 Starting Timestamp Analyzer"
echo "   Video dir: $VIDEO_DIR"
echo "   Open: http://localhost:$PORT"
echo ""

python frame_compare_web.py --dir "$VIDEO_DIR" --port "$PORT"
