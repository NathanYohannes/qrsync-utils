#!/usr/bin/env python3
"""
Sync Camera Setup - Web-based live camera preview with QR code detection.
Opens USB cameras (filtering out atlas/user-facing cameras) and displays
live feeds with real-time QR timestamp detection in a web browser.
"""

import argparse
import subprocess
import sys
import re
import time
import threading
import base64
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import uvicorn

# Try to import zxing-cpp for better QR detection
try:
    import zxingcpp
    ZXING_AVAILABLE = True
except ImportError:
    ZXING_AVAILABLE = False


@dataclass
class CameraInfo:
    """Information about a detected camera."""
    index: int
    name: str
    path: str
    width: int = 0
    height: int = 0
    fps: float = 0.0


class QRDetector:
    """QR code detector using multiple backends."""
    
    def __init__(self):
        self.opencv_detector = cv2.QRCodeDetector()
        self.wechat_detector = None
        try:
            self.wechat_detector = cv2.wechat_qrcode_WeChatQRCode()
            print("  Using WeChat QR detector (best for screens)")
        except Exception:
            print("  WeChat detector not available, using OpenCV fallback")
    
    def detect(self, frame: np.ndarray) -> Optional[str]:
        """Detect and decode QR code from frame."""
        if frame is None:
            return None
        
        # WeChat detector is best for screen-captured QR codes
        if self.wechat_detector:
            try:
                results, _ = self.wechat_detector.detectAndDecode(frame)
                if results and results[0]:
                    return results[0]
            except Exception:
                pass
        
        # Fall back to zxing-cpp
        if ZXING_AVAILABLE:
            try:
                results = zxingcpp.read_barcodes(frame)
                for result in results:
                    if result.format == zxingcpp.BarcodeFormat.QRCode and result.text:
                        return result.text
            except Exception:
                pass
        
        # Fall back to OpenCV detector
        try:
            data, _, _ = self.opencv_detector.detectAndDecode(frame)
            if data:
                return data
        except Exception:
            pass
        
        return None


# Patterns to filter out unwanted cameras
FILTER_PATTERNS = [
    r'atlas',
    r'facetime',
    r'iphone',
    r'ipad',
    r'continuity',
    r'virtual',
    r'obs',
    r'snap',
    r'zoom',
    r'teams',
    r'user.*facing',
    r'front.*camera',
    r'selfie',
]


def should_filter_camera(name: str) -> bool:
    """Check if camera should be filtered out based on name."""
    name_lower = name.lower()
    for pattern in FILTER_PATTERNS:
        if re.search(pattern, name_lower):
            return True
    return False


def is_capture_device(device_path: str) -> bool:
    """Check if a v4l2 device is a video capture device (not metadata)."""
    try:
        result = subprocess.run(
            ['v4l2-ctl', '-d', device_path, '--list-formats-ext'],
            capture_output=True, text=True, timeout=5
        )
        output = result.stdout
        if 'MJPG' in output or 'YUYV' in output or 'NV12' in output or 'H264' in output:
            return True
        return False
    except Exception:
        return True


def get_camera_name(device_path: str) -> str:
    """Get camera name using v4l2-ctl."""
    try:
        result = subprocess.run(
            ['v4l2-ctl', '-d', device_path, '--info'],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split('\n'):
            if 'Card type' in line:
                return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return "Unknown Camera"


def discover_cameras_linux() -> List[CameraInfo]:
    """Discover cameras on Linux using v4l2."""
    cameras = []
    name_counts = {}  # Track how many cameras have each name
    
    try:
        video_devices = sorted(Path('/dev').glob('video*'), 
                               key=lambda p: int(p.name.replace('video', '')) if p.name.replace('video', '').isdigit() else 999)
        
        for device in video_devices:
            try:
                idx = int(device.name.replace('video', ''))
            except ValueError:
                continue
            
            device_path = str(device)
            
            if not is_capture_device(device_path):
                continue
            
            name = get_camera_name(device_path)
            
            if should_filter_camera(name):
                print(f"  Filtered out: {name} ({device_path})")
                continue
            
            # Handle duplicate names by appending index
            if name in name_counts:
                name_counts[name] += 1
                display_name = f"{name} #{name_counts[name]}"
            else:
                name_counts[name] = 1
                display_name = name
            
            cameras.append(CameraInfo(index=idx, name=display_name, path=device_path))
                
    except Exception as e:
        print(f"Error discovering cameras: {e}")
    
    return cameras


def discover_cameras_macos() -> List[CameraInfo]:
    """Discover cameras on macOS."""
    cameras = []
    for idx in range(10):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cameras.append(CameraInfo(index=idx, name=f'Camera {idx}', path=str(idx)))
            cap.release()
    return cameras


def discover_cameras() -> List[CameraInfo]:
    """Discover available cameras."""
    if sys.platform == 'darwin':
        return discover_cameras_macos()
    else:
        return discover_cameras_linux()


class CameraManager:
    """Manages multiple camera feeds with QR detection."""
    
    def __init__(self):
        self.cameras: Dict[int, CameraInfo] = {}
        self.caps: Dict[int, cv2.VideoCapture] = {}
        self.detector = QRDetector()
        self.frames: Dict[int, Tuple[np.ndarray, Optional[str], float]] = {}
        self.lock = threading.Lock()
        self.running = False
        self.threads: List[threading.Thread] = []
    
    def open_cameras(self, camera_list: List[CameraInfo]) -> int:
        """Open cameras for capture."""
        opened = 0
        for cam in camera_list:
            if sys.platform == 'darwin':
                cap = cv2.VideoCapture(cam.index)
            else:
                cap = cv2.VideoCapture(cam.path)
            
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                cam.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                cam.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cam.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                
                self.caps[cam.index] = cap
                self.cameras[cam.index] = cam
                opened += 1
                print(f"  Opened: {cam.name} ({cam.width}x{cam.height})")
            else:
                print(f"  Failed: {cam.name} ({cam.path})")
        
        return opened
    
    def _capture_loop(self, cam_idx: int):
        """Capture loop for a camera."""
        cap = self.caps.get(cam_idx)
        if not cap:
            return
        
        while self.running:
            ret, frame = cap.read()
            if ret and frame is not None:
                qr_data = self.detector.detect(frame)
                with self.lock:
                    self.frames[cam_idx] = (frame.copy(), qr_data, time.time())
            time.sleep(0.01)
    
    def start(self):
        """Start capture threads."""
        self.running = True
        for cam_idx in self.caps:
            t = threading.Thread(target=self._capture_loop, args=(cam_idx,), daemon=True)
            t.start()
            self.threads.append(t)
    
    def stop(self):
        """Stop capture and release cameras."""
        self.running = False
        for t in self.threads:
            t.join(timeout=1.0)
        for cap in self.caps.values():
            cap.release()
        self.caps.clear()
        self.threads.clear()
    
    def get_frame_jpeg(self, cam_idx: int) -> Optional[bytes]:
        """Get latest frame as JPEG bytes."""
        with self.lock:
            if cam_idx not in self.frames:
                return None
            frame, _, _ = self.frames[cam_idx]
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return buf.tobytes()
    
    def get_status(self) -> Dict:
        """Get status of all cameras."""
        status = {}
        with self.lock:
            for idx, cam in self.cameras.items():
                qr_data = None
                if idx in self.frames:
                    _, qr_data, _ = self.frames[idx]
                status[idx] = {
                    'index': idx,
                    'name': cam.name,
                    'width': cam.width,
                    'height': cam.height,
                    'qr_detected': qr_data is not None,
                    'qr_data': qr_data,
                }
        return status


# Global
manager: Optional[CameraManager] = None


HTML_PAGE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sync Camera Setup</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;600&display=swap');
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-tertiary: #1a1a25;
            --accent: #00d4ff;
            --text-primary: #e8e8ed;
            --text-secondary: #8888a0;
            --success: #00ff88;
            --warning: #ffaa00;
            --danger: #ff4466;
            --border: #2a2a3a;
        }
        
        body {
            font-family: 'Space Grotesk', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container { max-width: 1800px; margin: 0 auto; }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 25px;
            background: var(--bg-secondary);
            border-radius: 12px;
            margin-bottom: 20px;
            border: 1px solid var(--border);
        }
        
        h1 {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--accent);
        }
        
        .camera-count {
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-secondary);
        }
        
        .camera-count span { color: var(--accent); }
        
        .camera-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(580px, 1fr));
            gap: 20px;
        }
        
        .camera-panel {
            background: var(--bg-secondary);
            border-radius: 12px;
            overflow: hidden;
            border: 2px solid var(--border);
            transition: all 0.3s ease;
        }
        
        .camera-panel.qr-detected {
            border-color: var(--success);
            box-shadow: 0 0 30px rgba(0, 255, 136, 0.3);
        }
        
        .camera-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 18px;
            background: var(--bg-tertiary);
        }
        
        .camera-title {
            font-weight: 600;
            color: var(--accent);
        }
        
        .qr-status {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .qr-indicator {
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: var(--danger);
            transition: all 0.2s ease;
            box-shadow: 0 0 10px var(--danger);
        }
        
        .qr-indicator.detected {
            background: var(--success);
            box-shadow: 0 0 15px var(--success);
        }
        
        .qr-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1rem;
            padding: 6px 14px;
            border-radius: 6px;
            background: rgba(0, 255, 136, 0.15);
            color: var(--success);
            min-width: 180px;
            text-align: center;
            font-weight: 600;
        }
        
        .qr-value.none {
            background: rgba(255, 68, 102, 0.15);
            color: var(--danger);
        }
        
        .frame-wrapper {
            position: relative;
            width: 100%;
            background: #000;
        }
        
        .frame-wrapper img {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .instructions {
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
            border: 1px solid var(--border);
        }
        
        .instructions h2 {
            font-size: 1rem;
            margin-bottom: 12px;
            color: var(--accent);
        }
        
        .instructions ul {
            list-style: none;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 10px;
        }
        
        .instructions li {
            font-size: 0.9rem;
            color: var(--text-secondary);
            padding: 10px 14px;
            background: var(--bg-tertiary);
            border-radius: 8px;
        }
        
        .instructions li strong { color: var(--success); }
        .instructions code { 
            background: var(--bg-primary); 
            padding: 2px 6px; 
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
        }
        
        .no-cameras {
            text-align: center;
            padding: 60px;
            background: var(--bg-secondary);
            border-radius: 12px;
            border: 1px solid var(--border);
        }
        
        .no-cameras h2 { color: var(--warning); margin-bottom: 15px; }
        .no-cameras p { color: var(--text-secondary); }
        
        @media (max-width: 700px) {
            .camera-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📷 Sync Camera Setup</h1>
            <div class="camera-count"><span id="cam-count">0</span> cameras</div>
        </header>
        
        <div class="camera-grid" id="grid">
            <div class="no-cameras">
                <h2>Loading cameras...</h2>
                <p>Please wait.</p>
            </div>
        </div>
        
        <div class="instructions">
            <h2>Setup Instructions</h2>
            <ul>
                <li><strong>1.</strong> Start the QR beacon: <code>./run_beacon.sh</code></li>
                <li><strong>2.</strong> Point cameras at the QR display</li>
                <li><strong>3.</strong> Adjust until all cameras show <strong>green</strong></li>
                <li><strong>4.</strong> Verify timestamps are updating</li>
                <li><strong>5.</strong> Start recording when ready</li>
            </ul>
        </div>
    </div>
    
    <script>
        let cameras = {};
        
        async function updateStatus() {
            try {
                const resp = await fetch('/api/status');
                const data = await resp.json();
                const camIds = Object.keys(data);
                
                document.getElementById('cam-count').textContent = camIds.length;
                
                if (camIds.length === 0) {
                    document.getElementById('grid').innerHTML = `
                        <div class="no-cameras">
                            <h2>No Cameras Found</h2>
                            <p>Make sure USB cameras are connected.</p>
                        </div>`;
                    return;
                }
                
                const grid = document.getElementById('grid');
                
                for (const id of camIds) {
                    const cam = data[id];
                    let panel = document.getElementById('panel-' + id);
                    
                    if (!panel) {
                        panel = document.createElement('div');
                        panel.id = 'panel-' + id;
                        panel.className = 'camera-panel';
                        panel.innerHTML = `
                            <div class="camera-header">
                                <div class="camera-title">Camera ${id}: ${cam.name}</div>
                                <div class="qr-status">
                                    <div class="qr-indicator" id="ind-${id}"></div>
                                    <div class="qr-value none" id="qr-${id}">No QR</div>
                                </div>
                            </div>
                            <div class="frame-wrapper">
                                <img id="img-${id}" src="/api/stream/${id}" alt="Camera ${id}">
                            </div>`;
                        grid.appendChild(panel);
                    }
                    
                    // Update QR status
                    const ind = document.getElementById('ind-' + id);
                    const qrEl = document.getElementById('qr-' + id);
                    
                    if (cam.qr_detected) {
                        ind.className = 'qr-indicator detected';
                        qrEl.className = 'qr-value';
                        qrEl.textContent = cam.qr_data;
                        panel.className = 'camera-panel qr-detected';
                    } else {
                        ind.className = 'qr-indicator';
                        qrEl.className = 'qr-value none';
                        qrEl.textContent = 'No QR';
                        panel.className = 'camera-panel';
                    }
                }
            } catch (e) {
                console.error('Status error:', e);
            }
        }
        
        // Poll status
        updateStatus();
        setInterval(updateStatus, 100);
    </script>
</body>
</html>
'''


@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    
    print("\nDiscovering cameras...")
    cameras = discover_cameras()
    
    if cameras:
        print(f"\nFound {len(cameras)} camera(s):")
        for c in cameras:
            print(f"  - [{c.index}] {c.name}")
        
        print("\nOpening cameras...")
        manager = CameraManager()
        opened = manager.open_cameras(cameras)
        
        if opened > 0:
            print(f"\nStarting capture ({opened} cameras)...")
            manager.start()
    else:
        print("\nNo cameras found!")
    
    yield
    
    print("\nShutting down...")
    if manager:
        manager.stop()


app = FastAPI(title="Sync Camera Setup", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.get("/api/status")
async def get_status():
    if not manager:
        return JSONResponse(content={})
    return JSONResponse(content=manager.get_status())


@app.get("/api/stream/{cam_id}")
async def stream(cam_id: int):
    """MJPEG stream for a camera."""
    if not manager:
        return JSONResponse(content={"error": "No cameras"}, status_code=404)
    
    def generate():
        while True:
            jpg = manager.get_frame_jpeg(cam_id)
            if jpg:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
            time.sleep(0.033)  # ~30fps
    
    return StreamingResponse(
        generate(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


def main():
    parser = argparse.ArgumentParser(
        description='Sync Camera Setup - Web-based camera preview with QR detection'
    )
    parser.add_argument('--port', type=int, default=5001, help='Port (default: 5001)')
    parser.add_argument('--host', default='0.0.0.0', help='Host (default: 0.0.0.0)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  SYNC CAMERA SETUP")
    print("  Web-based camera preview with QR detection")
    print("=" * 60)
    print(f"\n  Open: http://localhost:{args.port}")
    print(f"  Network: http://0.0.0.0:{args.port}")
    print("\n  Press Ctrl+C to stop\n")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == '__main__':
    main()
