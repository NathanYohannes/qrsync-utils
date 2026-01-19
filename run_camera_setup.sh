#!/bin/bash
# Quick-run script for Sync Camera Setup

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/sync_camera_setup"

if [ ! -d "venv" ]; then
    echo "Error: Run ./setup.sh first"
    exit 1
fi

source venv/bin/activate

PORT="${1:-5001}"

echo "📷 Starting Sync Camera Setup"
echo "   Open: http://localhost:$PORT"
echo ""

python sync_camera_setup.py --port "$PORT"
