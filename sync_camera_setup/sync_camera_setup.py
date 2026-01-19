#!/usr/bin/env python3
"""
Sync Camera Setup - Live camera preview with QR code detection.
Opens all USB cameras (filtering out atlas/user-facing cameras) and displays
live feeds with real-time QR timestamp detection.

Use this to verify cameras can see and decode the QR beacon before recording.
"""

import argparse
import subprocess
import sys
import base64
import time
import threading
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
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
    """QR code detector using multiple backends for best detection rate."""
    
    def __init__(self):
        self.opencv_detector = cv2.QRCodeDetector()
        self.wechat_detector = None
        try:
            self.wechat_detector = cv2.wechat_qrcode_WeChatQRCode()
        except Exception:
            pass
    
    def detect(self, frame: np.ndarray) -> Optional[str]:
        """Detect and decode QR code from frame. Returns decoded string or None."""
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


class CameraManager:
    """Manages multiple camera feeds with QR detection."""
    
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
    
    def __init__(self):
        self.cameras: Dict[int, cv2.VideoCapture] = {}
        self.camera_info: Dict[int, CameraInfo] = {}
        self.qr_detector = QRDetector()
        self.latest_frames: Dict[int, Tuple[np.ndarray, Optional[str], float]] = {}
        self.lock = threading.Lock()
        self.running = False
        self.threads: List[threading.Thread] = []
    
    def _should_filter_camera(self, name: str) -> bool:
        """Check if camera should be filtered out based on name."""
        name_lower = name.lower()
        for pattern in self.FILTER_PATTERNS:
            if re.search(pattern, name_lower):
                return True
        return False
    
    def _list_cameras_macos(self) -> List[CameraInfo]:
        """List cameras on macOS using system_profiler."""
        cameras = []
        try:
            result = subprocess.run(
                ['system_profiler', 'SPCameraDataType', '-json'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                if 'SPCameraDataType' in data:
                    for idx, cam in enumerate(data['SPCameraDataType']):
                        name = cam.get('_name', f'Camera {idx}')
                        cameras.append(CameraInfo(
                            index=idx,
                            name=name,
                            path=f'/dev/video{idx}'
                        ))
        except Exception:
            pass
        return cameras
    
    def _list_cameras_linux(self) -> List[CameraInfo]:
        """List cameras on Linux using v4l2."""
        cameras = []
        try:
            # List video devices
            video_devices = list(Path('/dev').glob('video*'))
            for device in sorted(video_devices):
                idx = int(device.name.replace('video', ''))
                name = f'Camera {idx}'
                
                # Try to get device name using v4l2-ctl
                try:
                    result = subprocess.run(
                        ['v4l2-ctl', '-d', str(device), '--info'],
                        capture_output=True, text=True, timeout=5
                    )
                    for line in result.stdout.split('\n'):
                        if 'Card type' in line:
                            name = line.split(':')[1].strip()
                            break
                except Exception:
                    pass
                
                cameras.append(CameraInfo(
                    index=idx,
                    name=name,
                    path=str(device)
                ))
        except Exception:
            pass
        return cameras
    
    def discover_cameras(self) -> List[CameraInfo]:
        """Discover available cameras, filtering out unwanted ones."""
        # Get camera list based on platform
        if sys.platform == 'darwin':
            cameras = self._list_cameras_macos()
        else:
            cameras = self._list_cameras_linux()
        
        # If platform-specific detection failed, probe indices
        if not cameras:
            for idx in range(10):
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    cameras.append(CameraInfo(
                        index=idx,
                        name=f'Camera {idx}',
                        path=f'/dev/video{idx}'
                    ))
                    cap.release()
        
        # Filter out unwanted cameras
        filtered = []
        for cam in cameras:
            if not self._should_filter_camera(cam.name):
                filtered.append(cam)
            else:
                print(f"  Filtered out: {cam.name}")
        
        return filtered
    
    def open_cameras(self, camera_indices: Optional[List[int]] = None) -> Dict[int, CameraInfo]:
        """Open cameras and start capture threads."""
        if camera_indices is None:
            cameras = self.discover_cameras()
            camera_indices = [c.index for c in cameras]
        
        opened = {}
        for idx in camera_indices:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                # Get camera info
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                
                # Try to set reasonable resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                # Re-read actual values
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                info = CameraInfo(
                    index=idx,
                    name=f'Camera {idx}',
                    path=f'/dev/video{idx}',
                    width=width,
                    height=height,
                    fps=fps
                )
                
                self.cameras[idx] = cap
                self.camera_info[idx] = info
                opened[idx] = info
                print(f"  Opened: Camera {idx} ({width}x{height} @ {fps:.1f}fps)")
            else:
                print(f"  Failed to open camera {idx}")
        
        return opened
    
    def _capture_loop(self, cam_idx: int):
        """Capture loop for a single camera."""
        cap = self.cameras.get(cam_idx)
        if not cap:
            return
        
        while self.running:
            ret, frame = cap.read()
            if ret and frame is not None:
                # Detect QR code
                qr_data = self.qr_detector.detect(frame)
                timestamp = time.time()
                
                with self.lock:
                    self.latest_frames[cam_idx] = (frame, qr_data, timestamp)
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.01)
    
    def start(self):
        """Start capture threads for all cameras."""
        self.running = True
        for cam_idx in self.cameras:
            thread = threading.Thread(target=self._capture_loop, args=(cam_idx,), daemon=True)
            thread.start()
            self.threads.append(thread)
    
    def stop(self):
        """Stop all capture threads and release cameras."""
        self.running = False
        for thread in self.threads:
            thread.join(timeout=1.0)
        for cap in self.cameras.values():
            cap.release()
        self.cameras.clear()
        self.threads.clear()
    
    def get_frame(self, cam_idx: int) -> Tuple[Optional[str], Optional[str], float]:
        """Get latest frame as base64 JPEG, QR data, and timestamp."""
        with self.lock:
            if cam_idx not in self.latest_frames:
                return None, None, 0
            
            frame, qr_data, timestamp = self.latest_frames[cam_idx]
            
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return img_base64, qr_data, timestamp
    
    def get_all_status(self) -> Dict[int, dict]:
        """Get status of all cameras."""
        status = {}
        with self.lock:
            for idx, info in self.camera_info.items():
                qr_data = None
                last_update = 0
                if idx in self.latest_frames:
                    _, qr_data, last_update = self.latest_frames[idx]
                
                status[idx] = {
                    'index': idx,
                    'name': info.name,
                    'width': info.width,
                    'height': info.height,
                    'fps': info.fps,
                    'qr_detected': qr_data is not None,
                    'qr_data': qr_data,
                    'last_update': last_update
                }
        return status


# Global camera manager
camera_manager: Optional[CameraManager] = None


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
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .status-badge {
            font-size: 0.75rem;
            padding: 4px 10px;
            border-radius: 20px;
            font-family: 'JetBrains Mono', monospace;
        }
        
        .status-badge.connected {
            background: rgba(0, 255, 136, 0.15);
            color: var(--success);
            border: 1px solid var(--success);
        }
        
        .status-badge.error {
            background: rgba(255, 68, 102, 0.15);
            color: var(--danger);
            border: 1px solid var(--danger);
        }
        
        .camera-count {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }
        
        .camera-count span { color: var(--accent); }
        
        .camera-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
        }
        
        .camera-panel {
            background: var(--bg-secondary);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border);
            transition: border-color 0.3s ease;
        }
        
        .camera-panel.qr-detected {
            border-color: var(--success);
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);
        }
        
        .camera-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 18px;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border);
        }
        
        .camera-title {
            font-weight: 600;
            font-size: 1rem;
            color: var(--accent);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .camera-meta {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: var(--text-secondary);
        }
        
        .qr-status {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .qr-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--danger);
            transition: background 0.2s ease;
        }
        
        .qr-indicator.detected { background: var(--success); }
        
        .qr-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            padding: 4px 10px;
            border-radius: 6px;
            background: rgba(0, 255, 136, 0.1);
            color: var(--success);
            min-width: 150px;
            text-align: center;
        }
        
        .qr-value.none {
            background: rgba(136, 136, 160, 0.1);
            color: var(--text-secondary);
        }
        
        .frame-wrapper {
            position: relative;
            width: 100%;
            aspect-ratio: 16/9;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .frame-wrapper img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        
        .no-frame {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }
        
        .overlay {
            position: absolute;
            bottom: 10px;
            left: 10px;
            right: 10px;
            display: flex;
            justify-content: space-between;
            pointer-events: none;
        }
        
        .overlay-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            padding: 4px 8px;
            border-radius: 4px;
            background: rgba(0, 0, 0, 0.7);
        }
        
        .overlay-badge.qr {
            color: var(--success);
            font-size: 1.2rem;
            font-weight: 600;
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
            color: var(--text-primary);
            margin-bottom: 12px;
        }
        
        .instructions ul {
            list-style: none;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 10px;
        }
        
        .instructions li {
            font-size: 0.85rem;
            color: var(--text-secondary);
            padding: 8px 12px;
            background: var(--bg-tertiary);
            border-radius: 6px;
        }
        
        .instructions li strong { color: var(--accent); }
        
        .no-cameras {
            text-align: center;
            padding: 60px 20px;
            background: var(--bg-secondary);
            border-radius: 12px;
            border: 1px solid var(--border);
        }
        
        .no-cameras h2 {
            font-size: 1.5rem;
            color: var(--warning);
            margin-bottom: 15px;
        }
        
        .no-cameras p {
            color: var(--text-secondary);
            margin-bottom: 10px;
        }
        
        @media (max-width: 600px) {
            .camera-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>
                <span>📷</span> Sync Camera Setup
                <span class="status-badge connected" id="connection-status">Connected</span>
            </h1>
            <div class="camera-count"><span id="camera-count">0</span> cameras detected</div>
        </header>
        
        <div class="camera-grid" id="camera-grid">
            <div class="no-cameras">
                <h2>Searching for cameras...</h2>
                <p>Please wait while we detect USB cameras.</p>
            </div>
        </div>
        
        <div class="instructions">
            <h2>Setup Instructions</h2>
            <ul>
                <li><strong>1.</strong> Start the QR beacon: <code>./run_beacon.sh</code></li>
                <li><strong>2.</strong> Point cameras at the QR display</li>
                <li><strong>3.</strong> Adjust until QR codes are detected (green indicator)</li>
                <li><strong>4.</strong> Verify timestamps are updating in real-time</li>
                <li><strong>5.</strong> Start recording when all cameras show green</li>
            </ul>
        </div>
    </div>
    
    <script>
        const cameras = {};
        let updateInterval = null;
        
        async function fetchStatus() {
            try {
                const resp = await fetch('/api/status');
                const data = await resp.json();
                
                document.getElementById('connection-status').className = 'status-badge connected';
                document.getElementById('connection-status').textContent = 'Connected';
                
                const cameraIds = Object.keys(data.cameras);
                document.getElementById('camera-count').textContent = cameraIds.length;
                
                if (cameraIds.length === 0) {
                    document.getElementById('camera-grid').innerHTML = `
                        <div class="no-cameras">
                            <h2>No USB Cameras Found</h2>
                            <p>Make sure cameras are connected and not in use by another application.</p>
                            <p>Atlas and user-facing cameras are automatically filtered out.</p>
                        </div>
                    `;
                    return;
                }
                
                // Create/update camera panels
                const grid = document.getElementById('camera-grid');
                
                for (const camId of cameraIds) {
                    const cam = data.cameras[camId];
                    let panel = document.getElementById(`cam-${camId}`);
                    
                    if (!panel) {
                        panel = document.createElement('div');
                        panel.id = `cam-${camId}`;
                        panel.className = 'camera-panel';
                        panel.innerHTML = `
                            <div class="camera-header">
                                <div>
                                    <div class="camera-title">📷 Camera ${camId}</div>
                                    <div class="camera-meta">${cam.width}x${cam.height} @ ${cam.fps.toFixed(0)}fps</div>
                                </div>
                                <div class="qr-status">
                                    <div class="qr-indicator" id="qr-ind-${camId}"></div>
                                    <div class="qr-value none" id="qr-val-${camId}">No QR</div>
                                </div>
                            </div>
                            <div class="frame-wrapper">
                                <img id="frame-${camId}" alt="Camera ${camId}" style="display: none;">
                                <div class="no-frame" id="no-frame-${camId}">Waiting for frame...</div>
                            </div>
                        `;
                        grid.appendChild(panel);
                    }
                    
                    // Update QR status
                    const indicator = document.getElementById(`qr-ind-${camId}`);
                    const valueEl = document.getElementById(`qr-val-${camId}`);
                    
                    if (cam.qr_detected && cam.qr_data) {
                        indicator.className = 'qr-indicator detected';
                        valueEl.className = 'qr-value';
                        valueEl.textContent = cam.qr_data;
                        panel.className = 'camera-panel qr-detected';
                    } else {
                        indicator.className = 'qr-indicator';
                        valueEl.className = 'qr-value none';
                        valueEl.textContent = 'No QR';
                        panel.className = 'camera-panel';
                    }
                }
                
                // Remove old panels
                const existingPanels = grid.querySelectorAll('.camera-panel');
                existingPanels.forEach(panel => {
                    const id = panel.id.replace('cam-', '');
                    if (!cameraIds.includes(id)) {
                        panel.remove();
                    }
                });
                
                // Fetch frames
                for (const camId of cameraIds) {
                    fetchFrame(camId);
                }
                
            } catch (e) {
                document.getElementById('connection-status').className = 'status-badge error';
                document.getElementById('connection-status').textContent = 'Disconnected';
            }
        }
        
        async function fetchFrame(camId) {
            try {
                const resp = await fetch(`/api/frame/${camId}`);
                const data = await resp.json();
                
                if (data.image) {
                    const img = document.getElementById(`frame-${camId}`);
                    const noFrame = document.getElementById(`no-frame-${camId}`);
                    
                    img.src = 'data:image/jpeg;base64,' + data.image;
                    img.style.display = 'block';
                    if (noFrame) noFrame.style.display = 'none';
                }
            } catch (e) {
                // Ignore frame fetch errors
            }
        }
        
        // Start polling
        fetchStatus();
        updateInterval = setInterval(fetchStatus, 100);  // 10 FPS update
    </script>
</body>
</html>
'''


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global camera_manager
    
    print("\nDiscovering cameras...")
    camera_manager = CameraManager()
    cameras = camera_manager.open_cameras()
    
    if cameras:
        print(f"\nStarting capture for {len(cameras)} camera(s)...")
        camera_manager.start()
    else:
        print("\nNo cameras found!")
    
    yield
    
    print("\nShutting down...")
    if camera_manager:
        camera_manager.stop()


app = FastAPI(title="Sync Camera Setup", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main page."""
    return HTML_PAGE


@app.get("/api/status")
async def get_status():
    """Get status of all cameras."""
    if not camera_manager:
        return JSONResponse(content={"cameras": {}})
    
    status = camera_manager.get_all_status()
    return JSONResponse(content={"cameras": status})


@app.get("/api/frame/{cam_id}")
async def get_frame(cam_id: int):
    """Get latest frame from a camera."""
    if not camera_manager:
        return JSONResponse(content={"error": "No camera manager"})
    
    image, qr_data, timestamp = camera_manager.get_frame(cam_id)
    return JSONResponse(content={
        "image": image,
        "qr_data": qr_data,
        "timestamp": timestamp
    })


def main():
    parser = argparse.ArgumentParser(
        description='Sync Camera Setup - Live camera preview with QR detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool helps you set up cameras for QR-based sync calibration.
It automatically detects USB cameras and filters out:
  - Atlas cameras
  - User-facing/FaceTime cameras
  - Virtual cameras (OBS, Snap, Zoom, etc.)

Usage:
  1. Start the QR beacon on another device/screen
  2. Run this tool to see live camera feeds
  3. Adjust cameras until QR codes are detected
  4. Start your recording when all cameras show green
        """
    )
    parser.add_argument('--port', type=int, default=5001, help='Port (default: 5001)')
    parser.add_argument('--host', default='127.0.0.1', help='Host (default: 127.0.0.1)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  SYNC CAMERA SETUP")
    print("  Live camera preview with QR detection")
    print("=" * 60)
    print(f"\n  Open: http://{args.host}:{args.port}")
    print("  Press Ctrl+C to stop\n")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == '__main__':
    main()
