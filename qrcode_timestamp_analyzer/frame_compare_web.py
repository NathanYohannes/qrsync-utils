#!/usr/bin/env python3
"""
FastAPI-based side-by-side frame comparison viewer for two camera videos.
Allows dynamic video selection through the web interface with drag-and-drop support.
"""

import argparse
import sys
import base64
import os
from pathlib import Path
from contextlib import asynccontextmanager
import threading

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# Try to import zxing-cpp for better QR detection
try:
    import zxingcpp
    ZXING_AVAILABLE = True
except ImportError:
    ZXING_AVAILABLE = False
    print("Warning: zxing-cpp not available, using OpenCV QR detector")

# Global state
viewer = None
video_lock = threading.Lock()
default_directory = None
scanned_videos = {}  # filename -> full_path mapping


class VideoLoadRequest(BaseModel):
    video1: str
    video2: str


class VideoFrameServer:
    def __init__(self, video1_path: str, video2_path: str):
        self.cap1 = cv2.VideoCapture(video1_path)
        self.cap2 = cv2.VideoCapture(video2_path)
        
        if not self.cap1.isOpened():
            raise ValueError(f"Could not open {video1_path}")
        if not self.cap2.isOpened():
            raise ValueError(f"Could not open {video2_path}")
        
        self.video1_path = video1_path
        self.video2_path = video2_path
        self.video1_name = Path(video1_path).name
        self.video2_name = Path(video2_path).name
        
        self.frame_count1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps1 = self.cap1.get(cv2.CAP_PROP_FPS) or 30.0
        self.fps2 = self.cap2.get(cv2.CAP_PROP_FPS) or 30.0
        self.width1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Count frames if metadata is unreliable (common with MJPEG containers)
        if self.frame_count1 <= 0 or self.frame_count1 > 10000000:
            print(f"Counting frames for {self.video1_name} (metadata unreliable)...")
            self.frame_count1 = self._count_frames(video1_path)
            # Reopen after counting
            self.cap1.release()
            self.cap1 = cv2.VideoCapture(video1_path)
        if self.frame_count2 <= 0 or self.frame_count2 > 10000000:
            print(f"Counting frames for {self.video2_name} (metadata unreliable)...")
            self.frame_count2 = self._count_frames(video2_path)
            # Reopen after counting
            self.cap2.release()
            self.cap2 = cv2.VideoCapture(video2_path)
        
        self.max_frames = min(self.frame_count1, self.frame_count2)
        
        self.qr_detector = cv2.QRCodeDetector()
        try:
            self.wechat_detector = cv2.wechat_qrcode_WeChatQRCode()
        except:
            self.wechat_detector = None
        
        self.frame_cache = {}
        self.cache_size = 100
    
    def _count_frames(self, video_path: str) -> int:
        """Count frames by reading through the video (slow but accurate)."""
        cap = cv2.VideoCapture(video_path)
        count = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            count += 1
            if count % 1000 == 0:
                print(f"  Counted {count} frames...")
        cap.release()
        print(f"  Total: {count} frames")
        return count
    
    def decode_qr(self, frame) -> str:
        if frame is None:
            return None
        
        # WeChat detector is best for screen-captured QR codes (handles moiré, blur, angles)
        if self.wechat_detector:
            try:
                results, _ = self.wechat_detector.detectAndDecode(frame)
                if results and results[0]:
                    return results[0]
            except:
                pass
        
        # Fall back to zxing-cpp
        if ZXING_AVAILABLE:
            try:
                results = zxingcpp.read_barcodes(frame)
                for result in results:
                    if result.format == zxingcpp.BarcodeFormat.QRCode and result.text:
                        return result.text
            except:
                pass
        
        # Fall back to OpenCV detector
        try:
            data, _, _ = self.qr_detector.detectAndDecode(frame)
            if data:
                return data
        except:
            pass
        
        return None
    
    def get_frame(self, cam: int, frame_num: int) -> tuple:
        cache_key = (cam, frame_num)
        
        if cache_key in self.frame_cache:
            return self.frame_cache[cache_key]
        
        with video_lock:
            cap = self.cap1 if cam == 1 else self.cap2
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            # Get PTS in microseconds (CAP_PROP_POS_MSEC gives milliseconds)
            pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            pts_us = int(pts_ms * 1000) if pts_ms > 0 else None
        
        if not ret or frame is None:
            return None, None, None
        
        timestamp = self.decode_qr(frame)
        
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        if len(self.frame_cache) > self.cache_size:
            keys_to_remove = list(self.frame_cache.keys())[:20]
            for k in keys_to_remove:
                del self.frame_cache[k]
        
        self.frame_cache[cache_key] = (img_base64, timestamp, pts_us)
        return img_base64, timestamp, pts_us
    
    def get_info(self):
        return {
            'video1_name': self.video1_name,
            'video2_name': self.video2_name,
            'video1_path': self.video1_path,
            'video2_path': self.video2_path,
            'max_frames': self.max_frames,
            'fps1': self.fps1,
            'fps2': self.fps2,
            'width1': self.width1,
            'height1': self.height1,
            'width2': self.width2,
            'height2': self.height2,
        }
    
    def close(self):
        self.cap1.release()
        self.cap2.release()


def scan_video_files(directory: str, recursive: bool = True) -> list:
    """Scan directory for video files."""
    global scanned_videos
    videos = []
    try:
        path = Path(directory)
        if path.exists():
            pattern = '**/*' if recursive else '*'
            for ext in ['.mkv', '.mp4', '.avi', '.mov']:
                for f in path.glob(f'{pattern}{ext}'):
                    full_path = str(f)
                    videos.append(full_path)
                    scanned_videos[f.name.lower()] = full_path
                for f in path.glob(f'{pattern}{ext.upper()}'):
                    full_path = str(f)
                    videos.append(full_path)
                    scanned_videos[f.name.lower()] = full_path
            videos = list(set(videos))
            videos.sort()
    except Exception as e:
        print(f"Error scanning directory: {e}")
    return videos


def find_video_by_name(filename: str) -> str:
    """Find a video's full path by its filename."""
    global scanned_videos
    key = filename.lower()
    if key in scanned_videos:
        return scanned_videos[key]
    # Try without extension matching
    for stored_name, path in scanned_videos.items():
        if key in stored_name or stored_name in key:
            return path
    return None


LANDING_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Frame Compare Viewer</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;600&display=swap');
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-tertiary: #1a1a25;
            --accent: #00d4ff;
            --accent-dim: #00a0cc;
            --text-primary: #e8e8ed;
            --text-secondary: #8888a0;
            --success: #00ff88;
            --warning: #ffaa00;
            --danger: #ff4466;
            --border: #2a2a3a;
            --cam1: #00aaff;
            --cam2: #ff6688;
        }
        
        body {
            font-family: 'Space Grotesk', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            width: 100%;
        }
        
        h1 {
            font-size: 2rem;
            font-weight: 600;
            color: var(--accent);
            text-align: center;
            margin-bottom: 10px;
        }
        
        .subtitle {
            text-align: center;
            color: var(--text-secondary);
            margin-bottom: 30px;
        }
        
        .drop-zones {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .drop-zone {
            background: var(--bg-secondary);
            border: 2px dashed var(--border);
            border-radius: 16px;
            padding: 30px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            position: relative;
        }
        
        .drop-zone:hover {
            border-color: var(--accent);
            background: var(--bg-tertiary);
        }
        
        .drop-zone.dragover {
            border-color: var(--accent);
            background: rgba(0, 212, 255, 0.1);
            transform: scale(1.02);
        }
        
        .drop-zone.cam1 { border-color: var(--cam1); }
        .drop-zone.cam1.has-file { background: rgba(0, 170, 255, 0.1); border-style: solid; }
        .drop-zone.cam2 { border-color: var(--cam2); }
        .drop-zone.cam2.has-file { background: rgba(255, 102, 136, 0.1); border-style: solid; }
        
        .drop-zone-label {
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 15px;
        }
        
        .cam1 .drop-zone-label { color: var(--cam1); }
        .cam2 .drop-zone-label { color: var(--cam2); }
        
        .drop-zone-icon {
            font-size: 2.5rem;
            margin-bottom: 15px;
            opacity: 0.7;
        }
        
        .drop-zone-text {
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-bottom: 10px;
        }
        
        .drop-zone-or {
            color: var(--text-secondary);
            font-size: 0.8rem;
            margin: 10px 0;
        }
        
        .drop-zone input[type="file"] {
            display: none;
        }
        
        .browse-btn {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            padding: 8px 16px;
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        
        .browse-btn:hover {
            border-color: var(--accent);
            color: var(--accent);
        }
        
        .selected-file {
            margin-top: 15px;
            padding: 10px;
            background: var(--bg-tertiary);
            border-radius: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            word-break: break-all;
        }
        
        .selected-file.found { color: var(--success); }
        .selected-file.not-found { color: var(--warning); }
        
        .card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 20px;
        }
        
        .card h2 {
            font-size: 1rem;
            color: var(--text-primary);
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .input-row {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        input[type="text"] {
            flex: 1;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            padding: 12px 16px;
            border: 1px solid var(--border);
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border-radius: 8px;
            outline: none;
        }
        
        input[type="text"]:focus {
            border-color: var(--accent);
        }
        
        .btn {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.9rem;
            font-weight: 500;
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        
        .btn-primary {
            background: var(--accent);
            color: var(--bg-primary);
            width: 100%;
        }
        
        .btn-primary:hover {
            background: var(--accent-dim);
        }
        
        .btn-primary:disabled {
            background: var(--border);
            color: var(--text-secondary);
            cursor: not-allowed;
        }
        
        .btn-secondary {
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border);
        }
        
        .btn-secondary:hover {
            border-color: var(--accent);
            color: var(--accent);
        }
        
        .error {
            background: rgba(255, 68, 102, 0.1);
            border: 1px solid var(--danger);
            color: var(--danger);
            padding: 12px 16px;
            border-radius: 8px;
            margin-top: 15px;
            font-size: 0.85rem;
            display: none;
        }
        
        .video-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 8px;
            max-height: 250px;
            overflow-y: auto;
            margin-top: 15px;
            padding: 5px;
        }
        
        .video-option {
            padding: 10px 12px;
            background: var(--bg-tertiary);
            border: 2px solid transparent;
            border-radius: 8px;
            cursor: pointer;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: var(--text-secondary);
            transition: all 0.15s ease;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .video-option:hover {
            border-color: var(--border);
            color: var(--text-primary);
        }
        
        .video-option.cam1 {
            border-color: var(--cam1);
            color: var(--cam1);
            background: rgba(0, 170, 255, 0.1);
        }
        
        .video-option.cam2 {
            border-color: var(--cam2);
            color: var(--cam2);
            background: rgba(255, 102, 136, 0.1);
        }
        
        .scan-status {
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: 10px;
        }
        
        .scan-status span { color: var(--accent); }
        
        .path-inputs {
            display: none;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
        }
        
        .path-inputs.show { display: block; }
        
        .path-input-group {
            margin-bottom: 15px;
        }
        
        .path-input-group label {
            display: block;
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }
        
        .path-input-group.cam1 label { color: var(--cam1); }
        .path-input-group.cam2 label { color: var(--cam2); }
        
        .instructions {
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-align: center;
            margin-top: 15px;
            line-height: 1.6;
        }
        
        .toggle-advanced {
            color: var(--accent);
            cursor: pointer;
            font-size: 0.85rem;
            text-align: center;
            margin-top: 15px;
        }
        
        .toggle-advanced:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ Frame Compare Viewer</h1>
        <p class="subtitle">Drop videos or browse to compare frame-by-frame</p>
        
        <div class="drop-zones">
            <div class="drop-zone cam1" id="dropzone1" onclick="document.getElementById('file1').click()">
                <div class="drop-zone-label">📷 Camera 1</div>
                <div class="drop-zone-icon">📂</div>
                <div class="drop-zone-text">Drop video here</div>
                <div class="drop-zone-or">— or —</div>
                <button class="browse-btn" onclick="event.stopPropagation(); document.getElementById('file1').click()">Browse Files</button>
                <input type="file" id="file1" accept=".mkv,.mp4,.avi,.mov" onchange="handleFileSelect(1, this)">
                <div class="selected-file" id="selected1" style="display: none;"></div>
            </div>
            
            <div class="drop-zone cam2" id="dropzone2" onclick="document.getElementById('file2').click()">
                <div class="drop-zone-label">📷 Camera 2</div>
                <div class="drop-zone-icon">📂</div>
                <div class="drop-zone-text">Drop video here</div>
                <div class="drop-zone-or">— or —</div>
                <button class="browse-btn" onclick="event.stopPropagation(); document.getElementById('file2').click()">Browse Files</button>
                <input type="file" id="file2" accept=".mkv,.mp4,.avi,.mov" onchange="handleFileSelect(2, this)">
                <div class="selected-file" id="selected2" style="display: none;"></div>
            </div>
        </div>
        
        <button class="btn btn-primary" id="compare-btn" onclick="loadVideos()" disabled>
            Select Both Videos to Compare
        </button>
        
        <div class="error" id="error"></div>
        
        <div class="card" style="margin-top: 20px;">
            <h2>📂 Scan Directory for Videos</h2>
            <div class="input-row">
                <input type="text" id="scan-dir" placeholder="/path/to/videos">
                <button class="btn btn-secondary" onclick="scanDirectory()">Scan</button>
            </div>
            <div class="scan-status" id="scan-status"></div>
            <div class="video-grid" id="video-grid"></div>
        </div>
        
        <div class="toggle-advanced" onclick="toggleAdvanced()">⚙️ Show manual path entry</div>
        
        <div class="path-inputs" id="path-inputs">
            <div class="path-input-group cam1">
                <label>Camera 1 Full Path</label>
                <input type="text" id="video1-path" placeholder="/full/path/to/camera1.mkv">
            </div>
            <div class="path-input-group cam2">
                <label>Camera 2 Full Path</label>
                <input type="text" id="video2-path" placeholder="/full/path/to/camera2.mkv">
            </div>
        </div>
        
        <p class="instructions">
            💡 <strong>Tip:</strong> First scan a directory, then drop or browse for videos.<br>
            The app will automatically find the full path for dropped files.
        </p>
    </div>
    
    <script>
        let selectedFiles = { 1: null, 2: null };
        let resolvedPaths = { 1: null, 2: null };
        
        // Initialize
        document.getElementById('scan-dir').value = 'DEFAULT_DIR';
        
        // Setup drag and drop
        ['dropzone1', 'dropzone2'].forEach((id, idx) => {
            const zone = document.getElementById(id);
            const camNum = idx + 1;
            
            zone.addEventListener('dragover', (e) => {
                e.preventDefault();
                zone.classList.add('dragover');
            });
            
            zone.addEventListener('dragleave', (e) => {
                e.preventDefault();
                zone.classList.remove('dragover');
            });
            
            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                zone.classList.remove('dragover');
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    handleDroppedFile(camNum, files[0]);
                }
            });
        });
        
        async function handleDroppedFile(camNum, file) {
            selectedFiles[camNum] = file.name;
            
            // Try to resolve the full path
            const resp = await fetch('/api/find?filename=' + encodeURIComponent(file.name));
            const data = await resp.json();
            
            const selectedEl = document.getElementById('selected' + camNum);
            const zoneEl = document.getElementById('dropzone' + camNum);
            
            if (data.path) {
                resolvedPaths[camNum] = data.path;
                selectedEl.textContent = '✓ ' + data.path;
                selectedEl.className = 'selected-file found';
                document.getElementById('video' + camNum + '-path').value = data.path;
            } else {
                resolvedPaths[camNum] = null;
                selectedEl.textContent = '⚠ ' + file.name + ' - Scan directory first';
                selectedEl.className = 'selected-file not-found';
            }
            
            selectedEl.style.display = 'block';
            zoneEl.classList.add('has-file');
            updateCompareButton();
        }
        
        async function handleFileSelect(camNum, input) {
            if (input.files.length > 0) {
                await handleDroppedFile(camNum, input.files[0]);
            }
        }
        
        function updateCompareButton() {
            const btn = document.getElementById('compare-btn');
            const path1 = resolvedPaths[1] || document.getElementById('video1-path').value.trim();
            const path2 = resolvedPaths[2] || document.getElementById('video2-path').value.trim();
            
            if (path1 && path2) {
                btn.disabled = false;
                btn.textContent = 'Compare Videos →';
            } else {
                btn.disabled = true;
                const missing = [];
                if (!path1) missing.push('Camera 1');
                if (!path2) missing.push('Camera 2');
                btn.textContent = 'Select ' + missing.join(' and ');
            }
        }
        
        async function loadVideos() {
            const path1 = resolvedPaths[1] || document.getElementById('video1-path').value.trim();
            const path2 = resolvedPaths[2] || document.getElementById('video2-path').value.trim();
            const errorEl = document.getElementById('error');
            
            if (!path1 || !path2) {
                errorEl.textContent = 'Please select both videos';
                errorEl.style.display = 'block';
                return;
            }
            
            errorEl.style.display = 'none';
            
            try {
                const resp = await fetch('/api/load', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ video1: path1, video2: path2 })
                });
                
                const data = await resp.json();
                
                if (resp.ok) {
                    window.location.href = '/viewer';
                } else {
                    errorEl.textContent = data.detail || 'Failed to load videos';
                    errorEl.style.display = 'block';
                }
            } catch (e) {
                errorEl.textContent = 'Error: ' + e.message;
                errorEl.style.display = 'block';
            }
        }
        
        async function scanDirectory() {
            const dir = document.getElementById('scan-dir').value.trim();
            if (!dir) return;
            
            const statusEl = document.getElementById('scan-status');
            statusEl.textContent = 'Scanning...';
            
            try {
                const resp = await fetch('/api/scan?directory=' + encodeURIComponent(dir));
                const data = await resp.json();
                
                const grid = document.getElementById('video-grid');
                
                if (data.videos && data.videos.length > 0) {
                    statusEl.innerHTML = 'Found <span>' + data.videos.length + '</span> videos. Click to select as Cam1/Cam2.';
                    
                    grid.innerHTML = data.videos.map(v => {
                        const name = v.split('/').pop();
                        const isSelected1 = resolvedPaths[1] === v;
                        const isSelected2 = resolvedPaths[2] === v;
                        const cls = isSelected1 ? 'cam1' : (isSelected2 ? 'cam2' : '');
                        return `<div class="video-option ${cls}" onclick="selectFromGrid('${v}')" title="${v}">${name}</div>`;
                    }).join('');
                    
                    // Try to resolve any pending dropped files
                    for (let cam of [1, 2]) {
                        if (selectedFiles[cam] && !resolvedPaths[cam]) {
                            const found = data.videos.find(v => v.toLowerCase().endsWith(selectedFiles[cam].toLowerCase()));
                            if (found) {
                                resolvedPaths[cam] = found;
                                const selectedEl = document.getElementById('selected' + cam);
                                selectedEl.textContent = '✓ ' + found;
                                selectedEl.className = 'selected-file found';
                                document.getElementById('video' + cam + '-path').value = found;
                                updateCompareButton();
                            }
                        }
                    }
                } else {
                    statusEl.textContent = 'No video files found in this directory.';
                    grid.innerHTML = '';
                }
            } catch (e) {
                statusEl.textContent = 'Error scanning: ' + e.message;
            }
        }
        
        function selectFromGrid(path) {
            // Toggle selection: first click = cam1, second click = cam2, third = deselect
            if (resolvedPaths[1] === path) {
                resolvedPaths[1] = null;
                document.getElementById('video1-path').value = '';
                document.getElementById('selected1').style.display = 'none';
                document.getElementById('dropzone1').classList.remove('has-file');
            } else if (resolvedPaths[2] === path) {
                resolvedPaths[2] = null;
                document.getElementById('video2-path').value = '';
                document.getElementById('selected2').style.display = 'none';
                document.getElementById('dropzone2').classList.remove('has-file');
            } else if (!resolvedPaths[1]) {
                resolvedPaths[1] = path;
                document.getElementById('video1-path').value = path;
                const el = document.getElementById('selected1');
                el.textContent = '✓ ' + path.split('/').pop();
                el.className = 'selected-file found';
                el.style.display = 'block';
                document.getElementById('dropzone1').classList.add('has-file');
            } else if (!resolvedPaths[2]) {
                resolvedPaths[2] = path;
                document.getElementById('video2-path').value = path;
                const el = document.getElementById('selected2');
                el.textContent = '✓ ' + path.split('/').pop();
                el.className = 'selected-file found';
                el.style.display = 'block';
                document.getElementById('dropzone2').classList.add('has-file');
            }
            
            updateCompareButton();
            refreshGridSelection();
        }
        
        function refreshGridSelection() {
            document.querySelectorAll('.video-option').forEach(el => {
                const path = el.getAttribute('title');
                el.classList.remove('cam1', 'cam2');
                if (resolvedPaths[1] === path) el.classList.add('cam1');
                if (resolvedPaths[2] === path) el.classList.add('cam2');
            });
        }
        
        function toggleAdvanced() {
            const el = document.getElementById('path-inputs');
            el.classList.toggle('show');
        }
        
        // Listen for manual path changes
        document.getElementById('video1-path').addEventListener('input', () => {
            resolvedPaths[1] = document.getElementById('video1-path').value.trim() || null;
            updateCompareButton();
        });
        document.getElementById('video2-path').addEventListener('input', () => {
            resolvedPaths[2] = document.getElementById('video2-path').value.trim() || null;
            updateCompareButton();
        });
        
        // Enter key support
        document.getElementById('video1-path').addEventListener('keypress', e => { if (e.key === 'Enter') loadVideos(); });
        document.getElementById('video2-path').addEventListener('keypress', e => { if (e.key === 'Enter') loadVideos(); });
        document.getElementById('scan-dir').addEventListener('keypress', e => { if (e.key === 'Enter') scanDirectory(); });
        
        // Initial scan
        if (document.getElementById('scan-dir').value && document.getElementById('scan-dir').value !== 'DEFAULT_DIR') {
            scanDirectory();
        }
    </script>
</body>
</html>
'''


VIEWER_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Frame Compare Viewer</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;600&display=swap');
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-tertiary: #1a1a25;
            --accent: #00d4ff;
            --accent-dim: #00a0cc;
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
            overflow-x: hidden;
        }
        
        .container { max-width: 1800px; margin: 0 auto; padding: 20px; }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 25px;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            margin-bottom: 20px;
            border-radius: 12px;
        }
        
        .header-left { display: flex; align-items: center; gap: 20px; }
        
        h1 { font-size: 1.4rem; font-weight: 600; color: var(--accent); }
        
        .back-btn {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            padding: 8px 14px;
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            border: 1px solid var(--border);
            border-radius: 6px;
            cursor: pointer;
            text-decoration: none;
            transition: all 0.15s ease;
        }
        
        .back-btn:hover { border-color: var(--accent); color: var(--accent); }
        
        .frame-info {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.1rem;
        }
        
        .frame-info span { color: var(--accent); }
        
        .video-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .video-panel {
            background: var(--bg-secondary);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border);
        }
        
        .video-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 18px;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border);
        }
        
        .video-title { font-weight: 600; font-size: 0.95rem; }
        .cam1 .video-title { color: #00aaff; }
        .cam2 .video-title { color: #ff6688; }
        
        .timestamp {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: var(--success);
            background: rgba(0, 255, 136, 0.1);
            padding: 4px 10px;
            border-radius: 6px;
        }
        
        .timestamp.none { color: var(--text-secondary); background: rgba(136, 136, 160, 0.1); }
        
        .frame-wrapper {
            position: relative;
            width: 100%;
            aspect-ratio: 4/3;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .frame-wrapper img { max-width: 100%; max-height: 100%; object-fit: contain; }
        
        .loading { position: absolute; color: var(--text-secondary); font-size: 0.9rem; }
        
        .drift-display {
            text-align: center;
            padding: 15px;
            background: var(--bg-secondary);
            border-radius: 12px;
            margin-bottom: 20px;
            border: 1px solid var(--border);
        }
        
        .drift-label { font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 5px; }
        
        .drift-value { font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 600; }
        .drift-value.good { color: var(--success); }
        .drift-value.warning { color: var(--warning); }
        .drift-value.bad { color: var(--danger); }
        .drift-value.none { color: var(--text-secondary); }
        
        .timeline-container {
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid var(--border);
        }
        
        .timeline-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        
        .timeline-slider {
            width: 100%;
            height: 8px;
            -webkit-appearance: none;
            background: var(--bg-tertiary);
            border-radius: 4px;
            outline: none;
            cursor: pointer;
        }
        
        .timeline-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: var(--accent);
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 0 15px rgba(0, 212, 255, 0.5);
        }
        
        .controls {
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        
        .btn {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            padding: 10px 18px;
            border: 1px solid var(--border);
            background: var(--bg-secondary);
            color: var(--text-primary);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        
        .btn:hover { background: var(--bg-tertiary); border-color: var(--accent); color: var(--accent); }
        .btn:active { transform: scale(0.97); }
        
        .btn.primary {
            background: var(--accent);
            color: var(--bg-primary);
            border-color: var(--accent);
            font-weight: 600;
        }
        
        .btn.primary:hover { background: var(--accent-dim); }
        
        .btn-group { display: flex; gap: 5px; }
        
        .goto-input {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            padding: 10px 14px;
            width: 120px;
            border: 1px solid var(--border);
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border-radius: 8px;
            outline: none;
        }
        
        .goto-input:focus { border-color: var(--accent); }
        
        .keyboard-hints {
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 18px;
            border: 1px solid var(--border);
        }
        
        .keyboard-hints h3 { font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 12px; }
        
        .hint-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 8px;
        }
        
        .hint { display: flex; align-items: center; gap: 10px; font-size: 0.85rem; }
        
        .key {
            font-family: 'JetBrains Mono', monospace;
            background: var(--bg-tertiary);
            padding: 4px 10px;
            border-radius: 5px;
            font-size: 0.8rem;
            color: var(--accent);
            border: 1px solid var(--border);
            min-width: 45px;
            text-align: center;
        }
        
        .hint-text { color: var(--text-secondary); }
        
        .status { font-size: 0.75rem; margin-left: 15px; }
        .status.loading { color: var(--warning); }
        .status.ready { color: var(--success); }
        
        @media (max-width: 1200px) { .video-container { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-left">
                <a href="/" class="back-btn">← New Videos</a>
                <h1>⚡ Frame Compare <span class="status ready" id="status">Ready</span></h1>
            </div>
            <div class="frame-info">Frame <span id="current-frame">0</span> / <span id="max-frames">0</span></div>
        </header>
        
        <div class="drift-display" style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
            <div>
                <div class="drift-label">PTS Offset (Cam2 - Cam1)</div>
                <div class="drift-value none" id="pts-offset" style="font-size: 1.4rem;">—</div>
            </div>
            <div>
                <div class="drift-label">QR Offset (Cam2 - Cam1)</div>
                <div class="drift-value none" id="qr-offset" style="font-size: 1.4rem;">—</div>
            </div>
            <div style="background: var(--bg-tertiary); padding: 10px; border-radius: 8px;">
                <div class="drift-label" style="color: var(--warning);">⚠ Sync Error (PTS - QR)</div>
                <div class="drift-value none" id="sync-error" style="font-size: 1.4rem;">—</div>
            </div>
        </div>
        
        <div class="video-container">
            <div class="video-panel cam1">
                <div class="video-header" style="flex-wrap: wrap; gap: 8px;">
                    <div class="video-title">📷 Cam1: <span id="video1-name">—</span></div>
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <div class="timestamp none" id="pts1" style="background: rgba(0, 170, 255, 0.2); color: #00aaff;">PTS: —</div>
                        <div class="timestamp none" id="ts1">QR: —</div>
                    </div>
                </div>
                <div class="frame-wrapper">
                    <div class="loading" id="loading1">Loading...</div>
                    <img id="frame1" alt="Camera 1">
                </div>
            </div>
            
            <div class="video-panel cam2">
                <div class="video-header" style="flex-wrap: wrap; gap: 8px;">
                    <div class="video-title">📷 Cam2: <span id="video2-name">—</span></div>
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <div class="timestamp none" id="pts2" style="background: rgba(255, 102, 136, 0.2); color: #ff6688;">PTS: —</div>
                        <div class="timestamp none" id="ts2">QR: —</div>
                    </div>
                </div>
                <div class="frame-wrapper">
                    <div class="loading" id="loading2">Loading...</div>
                    <img id="frame2" alt="Camera 2">
                </div>
            </div>
        </div>
        
        <div class="timeline-container">
            <div class="timeline-header">
                <span>0</span>
                <span id="timeline-time">00:00:00</span>
                <span id="max-frames-display">0</span>
            </div>
            <input type="range" class="timeline-slider" id="timeline" min="0" max="100" value="0">
        </div>
        
        <div class="controls">
            <div class="btn-group">
                <button class="btn" onclick="jump(-10000)">⏪ -10K</button>
                <button class="btn" onclick="jump(-1000)">◀◀ -1K</button>
                <button class="btn" onclick="jump(-100)">◀ -100</button>
                <button class="btn" onclick="jump(-10)">← -10</button>
            </div>
            
            <button class="btn" onclick="jump(-1)">‹ Prev</button>
            <button class="btn primary" onclick="togglePlay()" id="play-btn">▶ Play</button>
            <button class="btn" onclick="jump(1)">Next ›</button>
            
            <div class="btn-group">
                <button class="btn" onclick="jump(10)">+10 →</button>
                <button class="btn" onclick="jump(100)">+100 ▶</button>
                <button class="btn" onclick="jump(1000)">+1K ▶▶</button>
                <button class="btn" onclick="jump(10000)">+10K ⏩</button>
            </div>
            
            <div class="btn-group" style="margin-left: 20px;">
                <input type="number" class="goto-input" id="goto-input" placeholder="Frame #">
                <button class="btn" onclick="gotoFrame()">Go</button>
            </div>
        </div>
        
        <div class="keyboard-hints">
            <h3>⌨️ Keyboard Shortcuts</h3>
            <div class="hint-grid">
                <div class="hint"><span class="key">←</span><span class="hint-text">Previous frame</span></div>
                <div class="hint"><span class="key">→</span><span class="hint-text">Next frame</span></div>
                <div class="hint"><span class="key">↑</span><span class="hint-text">+10 frames</span></div>
                <div class="hint"><span class="key">↓</span><span class="hint-text">-10 frames</span></div>
                <div class="hint"><span class="key">PgUp</span><span class="hint-text">+100 frames</span></div>
                <div class="hint"><span class="key">PgDn</span><span class="hint-text">-100 frames</span></div>
                <div class="hint"><span class="key">Home</span><span class="hint-text">First frame</span></div>
                <div class="hint"><span class="key">End</span><span class="hint-text">Last frame</span></div>
                <div class="hint"><span class="key">Space</span><span class="hint-text">Play / Pause</span></div>
            </div>
        </div>
    </div>
    
    <script>
        let currentFrame = 0, maxFrames = 0, fps = 30, isPlaying = false, playInterval = null, pendingRequest = null;
        
        async function init() {
            try {
                const resp = await fetch('/api/info');
                if (!resp.ok) { window.location.href = '/'; return; }
                const info = await resp.json();
                
                maxFrames = info.max_frames;
                fps = info.fps1 || 30;
                
                document.getElementById('video1-name').textContent = info.video1_name;
                document.getElementById('video2-name').textContent = info.video2_name;
                document.getElementById('max-frames').textContent = maxFrames.toLocaleString();
                document.getElementById('max-frames-display').textContent = maxFrames.toLocaleString();
                document.getElementById('timeline').max = maxFrames - 1;
                
                loadFrame(0);
            } catch (e) { window.location.href = '/'; }
        }
        
        async function loadFrame(frameNum) {
            frameNum = Math.max(0, Math.min(frameNum, maxFrames - 1));
            currentFrame = frameNum;
            
            document.getElementById('current-frame').textContent = currentFrame.toLocaleString();
            document.getElementById('timeline').value = currentFrame;
            
            const seconds = Math.floor(currentFrame / fps);
            document.getElementById('timeline-time').textContent = 
                `${Math.floor(seconds/3600).toString().padStart(2,'0')}:${Math.floor((seconds%3600)/60).toString().padStart(2,'0')}:${(seconds%60).toString().padStart(2,'0')}`;
            
            const statusEl = document.getElementById('status');
            statusEl.textContent = 'Loading...';
            statusEl.className = 'status loading';
            
            if (pendingRequest) pendingRequest.abort();
            const controller = new AbortController();
            pendingRequest = controller;
            
            try {
                const [resp1, resp2] = await Promise.all([
                    fetch(`/api/frame/1/${frameNum}`, { signal: controller.signal }),
                    fetch(`/api/frame/2/${frameNum}`, { signal: controller.signal })
                ]);
                
                const data1 = await resp1.json();
                const data2 = await resp2.json();
                
                if (data1.image) { document.getElementById('frame1').src = 'data:image/jpeg;base64,' + data1.image; document.getElementById('loading1').style.display = 'none'; }
                if (data2.image) { document.getElementById('frame2').src = 'data:image/jpeg;base64,' + data2.image; document.getElementById('loading2').style.display = 'none'; }
                
                // Update PTS displays
                const pts1El = document.getElementById('pts1'), pts2El = document.getElementById('pts2');
                pts1El.textContent = data1.pts_us ? `PTS: ${(data1.pts_us/1000).toFixed(1)}ms` : 'PTS: —';
                pts1El.classList.toggle('none', !data1.pts_us);
                pts2El.textContent = data2.pts_us ? `PTS: ${(data2.pts_us/1000).toFixed(1)}ms` : 'PTS: —';
                pts2El.classList.toggle('none', !data2.pts_us);
                
                // Update QR displays
                const ts1El = document.getElementById('ts1'), ts2El = document.getElementById('ts2');
                ts1El.textContent = data1.timestamp ? `QR: ${data1.timestamp}` : 'QR: —';
                ts1El.classList.toggle('none', !data1.timestamp);
                ts2El.textContent = data2.timestamp ? `QR: ${data2.timestamp}` : 'QR: —';
                ts2El.classList.toggle('none', !data2.timestamp);
                
                // Calculate offsets
                const ptsOffsetEl = document.getElementById('pts-offset');
                const qrOffsetEl = document.getElementById('qr-offset');
                const syncErrorEl = document.getElementById('sync-error');
                
                let ptsOffset = null, qrOffset = null;
                
                // PTS Offset
                if (data1.pts_us && data2.pts_us) {
                    ptsOffset = (data2.pts_us - data1.pts_us) / 1000; // in ms
                    ptsOffsetEl.textContent = (ptsOffset >= 0 ? '+' : '') + ptsOffset.toFixed(1) + ' ms';
                    ptsOffsetEl.className = 'drift-value';
                    ptsOffsetEl.style.color = '#00aaff';
                } else {
                    ptsOffsetEl.textContent = '—';
                    ptsOffsetEl.className = 'drift-value none';
                }
                
                // QR Offset
                if (data1.timestamp && data2.timestamp) {
                    qrOffset = (parseInt(data2.timestamp) - parseInt(data1.timestamp)) / 1000; // in ms
                    qrOffsetEl.textContent = (qrOffset >= 0 ? '+' : '') + qrOffset.toFixed(1) + ' ms';
                    qrOffsetEl.className = 'drift-value ' + (Math.abs(qrOffset) < 20 ? 'good' : Math.abs(qrOffset) < 50 ? 'warning' : 'bad');
                } else {
                    qrOffsetEl.textContent = '—';
                    qrOffsetEl.className = 'drift-value none';
                }
                
                // Sync Error = PTS Offset - QR Offset
                // This shows how much the PTS is lying about real sync
                if (ptsOffset !== null && qrOffset !== null) {
                    const syncError = ptsOffset - qrOffset;
                    syncErrorEl.textContent = (syncError >= 0 ? '+' : '') + syncError.toFixed(1) + ' ms';
                    syncErrorEl.className = 'drift-value ' + (Math.abs(syncError) < 5 ? 'good' : Math.abs(syncError) < 20 ? 'warning' : 'bad');
                } else {
                    syncErrorEl.textContent = '—';
                    syncErrorEl.className = 'drift-value none';
                }
                
                statusEl.textContent = 'Ready';
                statusEl.className = 'status ready';
            } catch (e) { if (e.name !== 'AbortError') { statusEl.textContent = 'Error'; statusEl.className = 'status'; } }
            pendingRequest = null;
        }
        
        function jump(delta) { loadFrame(currentFrame + delta); }
        function gotoFrame() { const f = parseInt(document.getElementById('goto-input').value); if (!isNaN(f)) loadFrame(f); document.getElementById('goto-input').value = ''; }
        
        function togglePlay() {
            isPlaying = !isPlaying;
            document.getElementById('play-btn').textContent = isPlaying ? '⏸ Pause' : '▶ Play';
            if (isPlaying) {
                playInterval = setInterval(() => { if (currentFrame >= maxFrames - 1) togglePlay(); else loadFrame(currentFrame + 1); }, 1000 / fps);
            } else if (playInterval) { clearInterval(playInterval); playInterval = null; }
        }
        
        let timelineDebounce = null;
        document.getElementById('timeline').addEventListener('input', e => { if (timelineDebounce) clearTimeout(timelineDebounce); timelineDebounce = setTimeout(() => loadFrame(parseInt(e.target.value)), 50); });
        document.getElementById('goto-input').addEventListener('keypress', e => { if (e.key === 'Enter') gotoFrame(); });
        
        document.addEventListener('keydown', e => {
            if (e.target.tagName === 'INPUT') return;
            switch (e.key) {
                case 'ArrowLeft': e.preventDefault(); jump(-1); break;
                case 'ArrowRight': e.preventDefault(); jump(1); break;
                case 'ArrowUp': e.preventDefault(); jump(10); break;
                case 'ArrowDown': e.preventDefault(); jump(-10); break;
                case 'PageUp': e.preventDefault(); jump(100); break;
                case 'PageDown': e.preventDefault(); jump(-100); break;
                case 'Home': e.preventDefault(); loadFrame(0); break;
                case 'End': e.preventDefault(); loadFrame(maxFrames - 1); break;
                case ' ': e.preventDefault(); togglePlay(); break;
            }
        });
        
        init();
    </script>
</body>
</html>
'''


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if viewer:
        viewer.close()


app = FastAPI(title="Frame Compare Viewer", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    html = LANDING_HTML.replace('DEFAULT_DIR', default_directory or '/Volumes')
    return html


@app.get("/viewer", response_class=HTMLResponse)
async def viewer_page():
    if not viewer:
        return HTMLResponse(content="<script>window.location.href='/';</script>")
    return VIEWER_HTML


@app.get("/api/info")
async def get_info():
    if not viewer:
        raise HTTPException(status_code=404, detail="No videos loaded")
    return JSONResponse(content=viewer.get_info())


@app.get("/api/frame/{cam}/{frame_num}")
async def get_frame(cam: int, frame_num: int):
    if not viewer:
        raise HTTPException(status_code=404, detail="No videos loaded")
    if cam not in [1, 2]:
        raise HTTPException(status_code=400, detail="cam must be 1 or 2")
    
    image, timestamp, pts_us = viewer.get_frame(cam, frame_num)
    return JSONResponse(content={'image': image, 'timestamp': timestamp, 'pts_us': pts_us, 'frame': frame_num})


@app.post("/api/load")
async def load_videos(request: VideoLoadRequest):
    global viewer
    
    if viewer:
        viewer.close()
        viewer = None
    
    try:
        viewer = VideoFrameServer(request.video1, request.video2)
        return JSONResponse(content={"status": "ok", "info": viewer.get_info()})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load videos: {str(e)}")


@app.get("/api/scan")
async def scan_videos(directory: str):
    videos = scan_video_files(directory, recursive=True)
    return JSONResponse(content={"videos": videos, "directory": directory, "count": len(videos)})


@app.get("/api/find")
async def find_video(filename: str):
    """Find a video's full path by filename."""
    path = find_video_by_name(filename)
    return JSONResponse(content={"filename": filename, "path": path})


def main():
    global default_directory
    
    parser = argparse.ArgumentParser(description='FastAPI-based frame comparison viewer')
    parser.add_argument('--port', type=int, default=5000, help='Port (default: 5000)')
    parser.add_argument('--host', default='127.0.0.1', help='Host (default: 127.0.0.1)')
    parser.add_argument('--dir', default='/Volumes', help='Default scan directory')
    
    args = parser.parse_args()
    default_directory = args.dir
    
    # Pre-scan default directory
    if default_directory and Path(default_directory).exists():
        print(f"Pre-scanning {default_directory}...")
        videos = scan_video_files(default_directory, recursive=True)
        print(f"Found {len(videos)} videos")
    
    print("=" * 60)
    print("FRAME COMPARE WEB VIEWER")
    print("=" * 60)
    print(f"\n🚀 http://{args.host}:{args.port}")
    print("   Drag & drop videos or browse to select")
    print("   Press Ctrl+C to stop\n")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == '__main__':
    main()
