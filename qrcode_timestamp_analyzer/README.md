# QR Code Timestamp Analyzer

Compare two camera recordings frame-by-frame to analyze synchronization using QR code timestamps.

## Installation

```bash
cd qrcode_timestamp_analyzer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Tools

### 1. Web Viewer (`frame_compare_web.py`)

Interactive side-by-side frame comparison in your browser.

```bash
python frame_compare_web.py --dir "/path/to/videos"
```

Open http://127.0.0.1:5000 → drag & drop or browse for videos → compare frame-by-frame.

**Keyboard shortcuts:** ←/→ (±1), ↑/↓ (±10), PgUp/PgDn (±100), Space (play/pause)

---

### 2. Sync Analyzer (`analyze_sync.py`)

Analyze synchronization and generate timeline images.

```bash
# Basic usage - outputs to current directory
python analyze_sync.py cam1.mkv cam2.mkv

# Specify output location
python analyze_sync.py cam1.mkv cam2.mkv -o recordings/rec10/rec10

# More samples for accuracy
python analyze_sync.py cam1.mkv cam2.mkv -n 1000
```

**Outputs:**
- `*_timeline.png` - Detailed camera timelines with drift over time
- `*_overlay.png` - Overlaid comparison view
- `*.csv` - Raw data for further analysis

---

## Understanding Results

- **Drift** = Camera 2 timestamp − Camera 1 timestamp at same frame number
- **Positive drift**: Cam2 captured later than Cam1
- **Negative drift**: Cam2 captured earlier than Cam1
- **Large drift range (>1s)**: Indicates frame drops
- **Accumulating drift**: Camera frame rates don't match exactly
