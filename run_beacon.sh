#!/bin/bash
# Quick-run script for QR Timestamp Beacon

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/display_qrcode"

if [ ! -d "venv" ]; then
    echo "Error: Run ./setup.sh first"
    exit 1
fi

source venv/bin/activate

PORT="${1:-8080}"

echo "🔲 Starting QR Timestamp Beacon on port $PORT"
echo "   Open: http://localhost:$PORT"
echo ""

python main.py --port "$PORT"
