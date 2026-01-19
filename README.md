# QR Calibration Tools

A complete toolkit for camera synchronization analysis using QR code timestamps.

## Components

### 1. QR Timestamp Beacon (`display_qrcode/`)
Displays QR codes with embedded microsecond timestamps. Point your cameras at this display to record synchronization data.

### 2. Sync Camera Setup (`sync_camera_setup/`)
Live camera preview with real-time QR detection. Use this to verify cameras can see and decode QR codes before recording. Automatically filters out atlas cameras, FaceTime, and virtual cameras.

### 3. Timestamp Analyzer (`qrcode_timestamp_analyzer/`)
Analyzes recorded videos to compare frame timing between cameras and visualize synchronization drift.

## Quick Start

```bash
# 1. Extract the zip file (if downloaded)
unzip qr_calibration_tools.zip
cd qr_calibration_tools

# 2. Run setup (installs dependencies for all tools)
chmod +x *.sh
./setup.sh

# 3. Start the QR beacon (on the display device)
./run_beacon.sh

# 4. Verify camera setup (optional but recommended)
./run_camera_setup.sh

# 5. After recording, analyze with:
./run_analyzer.sh /path/to/videos
```

## Workflow

1. **Setup**: Run `./setup.sh` to install all dependencies

2. **Display QR Beacon**: 
   - Start the beacon: `./run_beacon.sh`
   - Open http://localhost:8080 on the display device

3. **Verify Camera Setup** (recommended):
   - Run: `./run_camera_setup.sh`
   - Open http://localhost:5001
   - Point cameras at the QR display
   - Verify all cameras show green (QR detected)
   - Adjust camera position/focus until detection is reliable

4. **Record**:
   - Point cameras at the QR code display
   - Start recording on all cameras simultaneously

5. **Analyze**:
   - Copy video files to the analysis device
   - Run: `./run_analyzer.sh /path/to/videos`
   - Open http://localhost:5000
   - Load both videos and compare frames

## Requirements

- Python 3.8+
- ~500MB disk space (for dependencies)
- Modern browser (Chrome/Firefox/Safari)
- USB cameras (for sync_camera_setup)

## Ports

- QR Beacon: `8080` (configurable: `./run_beacon.sh 9000`)
- Camera Setup: `5001` (configurable: `./run_camera_setup.sh 5002`)
- Analyzer: `5000` (configurable: `./run_analyzer.sh /videos 8000`)

## Camera Filtering

The Sync Camera Setup tool automatically filters out:
- Atlas cameras
- FaceTime / user-facing cameras
- Virtual cameras (OBS, Snap, Zoom, Teams)
- Continuity cameras (iPhone/iPad)

Only physical USB cameras are shown for sync calibration.
