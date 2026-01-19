#!/usr/bin/env python3
"""
PTS-Synchronized Frame Viewer

Synchronizes two video streams based on PTS (Presentation Time Stamps)
from MKV container timestamps. Shows frames that were captured at the
same moment in time, not just the same frame number.

Usage:
    python pts_sync_viewer.py <cam1.mkv> <cam2.mkv>
    python pts_sync_viewer.py  # Uses default rec12 paths
"""

import subprocess
import json
import sys
import os
import bisect
from typing import Optional, Tuple, List
from dataclasses import dataclass
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
import uvicorn


@dataclass
class FramePTS:
    """Frame index and its PTS timestamp in milliseconds"""
    frame_idx: int
    pts_ms: float


class PTSVideoReader:
    """Video reader that indexes frames by PTS timestamp"""
    
    def __init__(self, video_path: str):
        self.path = video_path
        self.cap: Optional[cv2.VideoCapture] = None
        self.pts_list: List[FramePTS] = []
        self.pts_to_frame: dict = {}  # pts_ms -> frame_idx
        self._current_frame = -1
        
    def open(self):
        """Open video and build PTS index"""
        print(f"Opening {os.path.basename(self.path)}...")
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.path}")
        
        # Build PTS index using ffprobe
        print(f"  Building PTS index...")
        self._build_pts_index()
        print(f"  Indexed {len(self.pts_list)} frames")
        print(f"  PTS range: {self.pts_list[0].pts_ms:.1f}ms - {self.pts_list[-1].pts_ms:.1f}ms")
        
    def _build_pts_index(self):
        """Extract PTS timestamps for all frames using ffprobe"""
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-select_streams', 'v:0',
            '-show_entries', 'packet=pts',
            '-of', 'csv=p=0',
            self.path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")
        
        self.pts_list = []
        for i, line in enumerate(result.stdout.strip().split('\n')):
            if line:
                pts_ms = float(line)
                self.pts_list.append(FramePTS(frame_idx=i, pts_ms=pts_ms))
                self.pts_to_frame[pts_ms] = i
    
    def get_pts_times(self) -> List[float]:
        """Return sorted list of PTS timestamps in ms"""
        return [f.pts_ms for f in self.pts_list]
    
    def find_frame_at_pts(self, target_pts_ms: float) -> int:
        """Find frame index closest to target PTS"""
        pts_times = self.get_pts_times()
        idx = bisect.bisect_left(pts_times, target_pts_ms)
        
        if idx == 0:
            return 0
        if idx >= len(pts_times):
            return len(self.pts_list) - 1
            
        # Check which is closer
        before = pts_times[idx - 1]
        after = pts_times[idx]
        
        if target_pts_ms - before <= after - target_pts_ms:
            return idx - 1
        return idx
    
    def read_frame(self, frame_idx: int) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame by index"""
        if frame_idx < 0 or frame_idx >= len(self.pts_list):
            return False, None
            
        if frame_idx != self._current_frame + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret, frame = self.cap.read()
        if ret:
            self._current_frame = frame_idx
        return ret, frame
    
    def read_frame_at_pts(self, pts_ms: float) -> Tuple[bool, Optional[np.ndarray], int, float]:
        """Read frame closest to PTS timestamp
        Returns: (success, frame, actual_frame_idx, actual_pts_ms)
        """
        frame_idx = self.find_frame_at_pts(pts_ms)
        ret, frame = self.read_frame(frame_idx)
        actual_pts = self.pts_list[frame_idx].pts_ms if frame_idx < len(self.pts_list) else 0
        return ret, frame, frame_idx, actual_pts
    
    @property
    def frame_count(self) -> int:
        return len(self.pts_list)
    
    @property
    def duration_ms(self) -> float:
        if not self.pts_list:
            return 0
        return self.pts_list[-1].pts_ms - self.pts_list[0].pts_ms
    
    @property
    def start_pts(self) -> float:
        return self.pts_list[0].pts_ms if self.pts_list else 0
    
    @property
    def end_pts(self) -> float:
        return self.pts_list[-1].pts_ms if self.pts_list else 0
    
    def close(self):
        if self.cap:
            self.cap.release()
            self.cap = None


class PTSSyncServer:
    """Web server for PTS-synchronized frame viewing"""
    
    def __init__(self, cam1_path: str, cam2_path: str):
        self.cam1 = PTSVideoReader(cam1_path)
        self.cam2 = PTSVideoReader(cam2_path)
        self.current_pts_ms: float = 0
        
    def open(self):
        self.cam1.open()
        self.cam2.open()
        
        # Calculate sync info
        self.pts_offset = self.cam2.start_pts - self.cam1.start_pts
        
        # Common timeline (intersection of both videos)
        self.common_start = max(self.cam1.start_pts, self.cam2.start_pts)
        self.common_end = min(self.cam1.end_pts, self.cam2.end_pts)
        self.common_duration = self.common_end - self.common_start
        
        print(f"\n=== PTS SYNC INFO ===")
        print(f"CAM1: {self.cam1.frame_count} frames, {self.cam1.start_pts:.1f} - {self.cam1.end_pts:.1f} ms")
        print(f"CAM2: {self.cam2.frame_count} frames, {self.cam2.start_pts:.1f} - {self.cam2.end_pts:.1f} ms")
        print(f"PTS Offset (cam2 - cam1): {self.pts_offset:.1f} ms")
        print(f"Common timeline: {self.common_start:.1f} - {self.common_end:.1f} ms ({self.common_duration/1000:.1f}s)")
        
        self.current_pts_ms = self.common_start
        
    def close(self):
        self.cam1.close()
        self.cam2.close()
        
    def get_synced_frames(self, pts_ms: float) -> dict:
        """Get frames from both cameras at the given PTS timestamp"""
        ret1, frame1, idx1, pts1 = self.cam1.read_frame_at_pts(pts_ms)
        ret2, frame2, idx2, pts2 = self.cam2.read_frame_at_pts(pts_ms)
        
        return {
            'cam1': {'frame': frame1, 'frame_idx': idx1, 'pts_ms': pts1, 'ok': ret1},
            'cam2': {'frame': frame2, 'frame_idx': idx2, 'pts_ms': pts2, 'ok': ret2},
            'target_pts_ms': pts_ms,
            'pts_diff_ms': abs(pts1 - pts2) if ret1 and ret2 else None
        }


# Global server instance
server: Optional[PTSSyncServer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global server
    if server:
        server.open()
    yield
    if server:
        server.close()


app = FastAPI(lifespan=lifespan)


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>PTS Sync Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            background: #0a0a0f;
            color: #e0e0e0;
            font-family: 'JetBrains Mono', 'SF Mono', 'Consolas', monospace;
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(180deg, #1a1a24 0%, #12121a 100%);
            padding: 12px 20px;
            border-bottom: 1px solid #2a2a3a;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .title {
            font-size: 18px;
            font-weight: 600;
            color: #00d4aa;
        }
        
        .sync-info {
            display: flex;
            gap: 20px;
            font-size: 13px;
        }
        
        .sync-info span {
            color: #888;
        }
        
        .sync-info .value {
            color: #00d4aa;
            font-weight: 500;
        }
        
        .pts-diff {
            padding: 4px 10px;
            border-radius: 4px;
            font-weight: 600;
        }
        
        .pts-diff.good { background: #0a3d2a; color: #00ff88; }
        .pts-diff.warn { background: #3d3a0a; color: #ffdd00; }
        .pts-diff.bad { background: #3d0a0a; color: #ff4444; }
        
        .video-container {
            flex: 1;
            display: flex;
            gap: 4px;
            padding: 10px;
            min-height: 0;
        }
        
        .video-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #12121a;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #2a2a3a;
        }
        
        .video-label {
            padding: 8px 12px;
            background: #1a1a24;
            border-bottom: 1px solid #2a2a3a;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .cam-name {
            font-weight: 600;
            font-size: 14px;
        }
        
        .cam-name.cam1 { color: #00aaff; }
        .cam-name.cam2 { color: #ff6600; }
        
        .frame-info {
            font-size: 12px;
            color: #888;
        }
        
        .frame-info .pts {
            color: #00d4aa;
        }
        
        .video-frame {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 0;
            padding: 8px;
        }
        
        .video-frame img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            border-radius: 4px;
        }
        
        .controls {
            background: linear-gradient(0deg, #1a1a24 0%, #12121a 100%);
            padding: 15px 20px;
            border-top: 1px solid #2a2a3a;
        }
        
        .timeline-container {
            margin-bottom: 12px;
        }
        
        .timeline-labels {
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: #666;
            margin-bottom: 4px;
        }
        
        .timeline {
            width: 100%;
            height: 8px;
            -webkit-appearance: none;
            appearance: none;
            background: #2a2a3a;
            border-radius: 4px;
            cursor: pointer;
        }
        
        .timeline::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            background: #00d4aa;
            border-radius: 50%;
            cursor: grab;
            box-shadow: 0 0 10px #00d4aa55;
        }
        
        .button-row {
            display: flex;
            justify-content: center;
            gap: 8px;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 8px 16px;
            background: #2a2a3a;
            border: 1px solid #3a3a4a;
            border-radius: 6px;
            color: #e0e0e0;
            font-family: inherit;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.15s;
        }
        
        .btn:hover {
            background: #3a3a4a;
            border-color: #00d4aa;
        }
        
        .btn:active {
            transform: scale(0.97);
        }
        
        .btn.primary {
            background: #00d4aa22;
            border-color: #00d4aa;
            color: #00d4aa;
        }
        
        .time-display {
            text-align: center;
            font-size: 24px;
            font-weight: 600;
            color: #00d4aa;
            margin-bottom: 10px;
            font-variant-numeric: tabular-nums;
        }
        
        .keyboard-hints {
            text-align: center;
            font-size: 11px;
            color: #555;
            margin-top: 10px;
        }
        
        .keyboard-hints kbd {
            background: #2a2a3a;
            padding: 2px 6px;
            border-radius: 3px;
            margin: 0 2px;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="title">⏱️ PTS Sync Viewer</div>
        <div class="sync-info">
            <div>
                <span>Duration:</span>
                <span class="value" id="duration">--</span>
            </div>
            <div>
                <span>PTS Offset:</span>
                <span class="value" id="offset">--</span>
            </div>
            <div id="pts-diff-container">
                <span>Frame Δ:</span>
                <span class="pts-diff good" id="pts-diff">--</span>
            </div>
        </div>
    </div>
    
    <div class="video-container">
        <div class="video-panel">
            <div class="video-label">
                <span class="cam-name cam1">CAM1</span>
                <span class="frame-info">
                    Frame <span id="cam1-frame">--</span> | 
                    PTS <span class="pts" id="cam1-pts">--</span>
                </span>
            </div>
            <div class="video-frame">
                <img id="img1" src="" alt="CAM1">
            </div>
        </div>
        
        <div class="video-panel">
            <div class="video-label">
                <span class="cam-name cam2">CAM2</span>
                <span class="frame-info">
                    Frame <span id="cam2-frame">--</span> | 
                    PTS <span class="pts" id="cam2-pts">--</span>
                </span>
            </div>
            <div class="video-frame">
                <img id="img2" src="" alt="CAM2">
            </div>
        </div>
    </div>
    
    <div class="controls">
        <div class="time-display" id="current-time">00:00.000</div>
        
        <div class="timeline-container">
            <div class="timeline-labels">
                <span id="start-time">0:00</span>
                <span id="end-time">--:--</span>
            </div>
            <input type="range" class="timeline" id="timeline" min="0" max="1000" value="0">
        </div>
        
        <div class="button-row">
            <button class="btn" onclick="jump(-10000)">-10s</button>
            <button class="btn" onclick="jump(-1000)">-1s</button>
            <button class="btn" onclick="jump(-100)">-100ms</button>
            <button class="btn" onclick="stepFrame(-1)">◀ Prev</button>
            <button class="btn primary" onclick="togglePlay()" id="play-btn">▶ Play</button>
            <button class="btn" onclick="stepFrame(1)">Next ▶</button>
            <button class="btn" onclick="jump(100)">+100ms</button>
            <button class="btn" onclick="jump(1000)">+1s</button>
            <button class="btn" onclick="jump(10000)">+10s</button>
        </div>
        
        <div class="keyboard-hints">
            <kbd>←</kbd><kbd>→</kbd> Step frame
            <kbd>Shift</kbd>+<kbd>←</kbd><kbd>→</kbd> ±1s
            <kbd>Space</kbd> Play/Pause
            <kbd>Home</kbd><kbd>End</kbd> Start/End
        </div>
    </div>

    <script>
        let info = null;
        let currentPts = 0;
        let playing = false;
        let playInterval = null;
        let avgFrameDuration = 16.67;  // ~60fps default
        
        async function init() {
            const resp = await fetch('/info');
            info = await resp.json();
            
            currentPts = info.common_start;
            avgFrameDuration = info.common_duration / Math.max(info.cam1_frames, info.cam2_frames);
            
            document.getElementById('duration').textContent = formatTime(info.common_duration);
            document.getElementById('offset').textContent = info.pts_offset.toFixed(1) + 'ms';
            document.getElementById('end-time').textContent = formatTimeShort(info.common_duration);
            
            const timeline = document.getElementById('timeline');
            timeline.min = info.common_start;
            timeline.max = info.common_end;
            timeline.value = currentPts;
            
            loadFrames();
        }
        
        function formatTime(ms) {
            const totalSec = ms / 1000;
            const min = Math.floor(totalSec / 60);
            const sec = totalSec % 60;
            return `${min}:${sec.toFixed(3).padStart(6, '0')}`;
        }
        
        function formatTimeShort(ms) {
            const totalSec = ms / 1000;
            const min = Math.floor(totalSec / 60);
            const sec = Math.floor(totalSec % 60);
            return `${min}:${sec.toString().padStart(2, '0')}`;
        }
        
        async function loadFrames() {
            const ts = Date.now();
            
            // Update UI immediately
            document.getElementById('current-time').textContent = formatTime(currentPts - info.common_start);
            document.getElementById('timeline').value = currentPts;
            
            // Load images
            document.getElementById('img1').src = `/frame/1?pts=${currentPts}&_=${ts}`;
            document.getElementById('img2').src = `/frame/2?pts=${currentPts}&_=${ts}`;
            
            // Get frame info
            const resp = await fetch(`/frame_info?pts=${currentPts}`);
            const data = await resp.json();
            
            document.getElementById('cam1-frame').textContent = data.cam1_frame;
            document.getElementById('cam1-pts').textContent = data.cam1_pts.toFixed(1) + 'ms';
            document.getElementById('cam2-frame').textContent = data.cam2_frame;
            document.getElementById('cam2-pts').textContent = data.cam2_pts.toFixed(1) + 'ms';
            
            // PTS difference indicator
            const diff = data.pts_diff;
            const diffEl = document.getElementById('pts-diff');
            diffEl.textContent = diff.toFixed(1) + 'ms';
            diffEl.className = 'pts-diff ' + (diff < 5 ? 'good' : diff < 20 ? 'warn' : 'bad');
        }
        
        function jump(deltaMs) {
            currentPts = Math.max(info.common_start, Math.min(info.common_end, currentPts + deltaMs));
            loadFrames();
        }
        
        function stepFrame(delta) {
            // Step by approximate frame duration
            jump(delta * avgFrameDuration);
        }
        
        function seekTo(pts) {
            currentPts = parseFloat(pts);
            loadFrames();
        }
        
        function togglePlay() {
            playing = !playing;
            document.getElementById('play-btn').textContent = playing ? '⏸ Pause' : '▶ Play';
            
            if (playing) {
                playInterval = setInterval(() => {
                    if (currentPts >= info.common_end) {
                        togglePlay();
                        return;
                    }
                    jump(avgFrameDuration);
                }, avgFrameDuration);
            } else {
                clearInterval(playInterval);
            }
        }
        
        // Timeline scrubbing
        const timeline = document.getElementById('timeline');
        let scrubbing = false;
        
        timeline.addEventListener('mousedown', () => { scrubbing = true; });
        timeline.addEventListener('mouseup', () => { scrubbing = false; seekTo(timeline.value); });
        timeline.addEventListener('input', () => {
            if (scrubbing) {
                currentPts = parseFloat(timeline.value);
                document.getElementById('current-time').textContent = formatTime(currentPts - info.common_start);
            }
        });
        
        // Keyboard controls
        document.addEventListener('keydown', (e) => {
            if (e.code === 'ArrowLeft') {
                e.preventDefault();
                e.shiftKey ? jump(-1000) : stepFrame(-1);
            } else if (e.code === 'ArrowRight') {
                e.preventDefault();
                e.shiftKey ? jump(1000) : stepFrame(1);
            } else if (e.code === 'Space') {
                e.preventDefault();
                togglePlay();
            } else if (e.code === 'Home') {
                e.preventDefault();
                currentPts = info.common_start;
                loadFrames();
            } else if (e.code === 'End') {
                e.preventDefault();
                currentPts = info.common_end;
                loadFrames();
            }
        });
        
        init();
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.get("/info")
async def get_info():
    return {
        "common_start": server.common_start,
        "common_end": server.common_end,
        "common_duration": server.common_duration,
        "pts_offset": server.pts_offset,
        "cam1_frames": server.cam1.frame_count,
        "cam2_frames": server.cam2.frame_count,
    }


@app.get("/frame/{cam}")
async def get_frame(cam: int, pts: float):
    reader = server.cam1 if cam == 1 else server.cam2
    ret, frame, idx, actual_pts = reader.read_frame_at_pts(pts)
    
    if not ret or frame is None:
        return Response(content=b'', media_type='image/jpeg')
    
    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(content=jpeg.tobytes(), media_type='image/jpeg')


@app.get("/frame_info")
async def get_frame_info(pts: float):
    ret1, _, idx1, pts1 = server.cam1.read_frame_at_pts(pts)
    ret2, _, idx2, pts2 = server.cam2.read_frame_at_pts(pts)
    
    return {
        "cam1_frame": idx1,
        "cam1_pts": pts1,
        "cam2_frame": idx2,
        "cam2_pts": pts2,
        "pts_diff": abs(pts1 - pts2) if ret1 and ret2 else 0
    }


def main():
    global server
    
    # Default paths for rec12
    default_cam1 = "/Volumes/Dev4/development/rec12/cam1_rec12.mkv"
    default_cam2 = "/Volumes/Dev4/development/rec12/cam2_rec12.mkv"
    
    if len(sys.argv) >= 3:
        cam1_path = sys.argv[1]
        cam2_path = sys.argv[2]
    else:
        cam1_path = default_cam1
        cam2_path = default_cam2
    
    print("=" * 60)
    print("PTS SYNC VIEWER")
    print("=" * 60)
    print(f"CAM1: {cam1_path}")
    print(f"CAM2: {cam2_path}")
    print()
    
    server = PTSSyncServer(cam1_path, cam2_path)
    
    print("\n🚀 http://127.0.0.1:5001")
    print("   Frames synchronized by PTS timestamps")
    print("   Press Ctrl+C to stop")
    
    uvicorn.run(app, host="127.0.0.1", port=5001, log_level="warning")


if __name__ == "__main__":
    main()
