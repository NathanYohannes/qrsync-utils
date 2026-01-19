# QR Calibration Tools

A toolkit for camera synchronization analysis using QR code timestamps.

## Components

### 1. QR Timestamp Beacon (`display_qrcode/`)
Displays QR codes with embedded millisecond timestamps. Point your cameras at this display to record synchronization data.

### 2. Timestamp Analyzer (`qrcode_timestamp_analyzer/`)
Analyzes recorded videos to compare frame timing between cameras and visualize synchronization drift.

## Quick Start

```bash
# 1. Extract the zip file
unzip qr_calibration_tools.zip
cd qr_calibration_tools

# 2. Run setup (installs dependencies for both tools)
chmod +x *.sh
./setup.sh

# 3. Start the QR beacon (on the display device)
./run_beacon.sh

# 4. After recording, analyze with:
./run_analyzer.sh /path/to/videos
```

## Workflow

1. **Setup**: Run `./setup.sh` on both devices (display device and analysis device)

2. **Record**: 
   - Start the beacon: `./run_beacon.sh`
   - Open http://localhost:8080 on the display
   - Point cameras at the QR code display
   - Record simultaneously

3. **Analyze**:
   - Copy video files to the analysis device
   - Run: `./run_analyzer.sh /path/to/videos`
   - Open http://localhost:5000
   - Load both videos and compare frames

## Requirements

- Python 3.8+
- ~500MB disk space (for dependencies)
- Modern browser (Chrome/Firefox/Safari)

## Ports

- QR Beacon: `8080` (configurable: `./run_beacon.sh 9000`)
- Analyzer: `5000` (configurable: `./run_analyzer.sh /videos 8000`)
