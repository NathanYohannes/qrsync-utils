#!/usr/bin/env python3
"""
Sync Camera Setup - Live camera preview with QR code detection.
Opens USB cameras (filtering out atlas/user-facing cameras) and displays
live feeds with real-time QR timestamp detection.

Two modes:
  - Default: OpenCV windows with QR detection overlay
  - --ffplay: Uses ffplay for display (no QR detection, but more compatible)
"""

import argparse
import subprocess
import sys
import re
import time
import threading
import signal
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

import cv2
import numpy as np

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


class QRDetector:
    """QR code detector using multiple backends."""
    
    def __init__(self):
        self.opencv_detector = cv2.QRCodeDetector()
        self.wechat_detector = None
        try:
            self.wechat_detector = cv2.wechat_qrcode_WeChatQRCode()
        except Exception:
            pass
    
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
        # Check for actual video formats
        if 'MJPG' in output or 'YUYV' in output or 'NV12' in output or 'H264' in output:
            return True
        return False
    except Exception:
        # If we can't check, assume it might be valid
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
    seen_names = {}  # Track cameras by name to avoid duplicates
    
    try:
        video_devices = sorted(Path('/dev').glob('video*'), 
                               key=lambda p: int(p.name.replace('video', '')) if p.name.replace('video', '').isdigit() else 999)
        
        for device in video_devices:
            try:
                idx = int(device.name.replace('video', ''))
            except ValueError:
                continue
            
            device_path = str(device)
            
            # Check if it's actually a capture device
            if not is_capture_device(device_path):
                continue
            
            name = get_camera_name(device_path)
            
            # Skip if we already have a camera with this name (avoid duplicates)
            # Keep the lower-numbered device
            if name in seen_names:
                continue
            seen_names[name] = idx
            
            if not should_filter_camera(name):
                cameras.append(CameraInfo(
                    index=idx,
                    name=name,
                    path=device_path
                ))
            else:
                print(f"  Filtered out: {name} ({device_path})")
                
    except Exception as e:
        print(f"Error discovering cameras: {e}")
    
    return cameras


def discover_cameras_macos() -> List[CameraInfo]:
    """Discover cameras on macOS."""
    cameras = []
    for idx in range(10):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            name = f'Camera {idx}'
            cameras.append(CameraInfo(
                index=idx,
                name=name,
                path=str(idx)
            ))
            cap.release()
    return cameras


def discover_cameras() -> List[CameraInfo]:
    """Discover available cameras."""
    if sys.platform == 'darwin':
        return discover_cameras_macos()
    else:
        return discover_cameras_linux()


def launch_ffplay(camera: CameraInfo) -> subprocess.Popen:
    """Launch ffplay for a camera."""
    window_title = f"Camera {camera.index}: {camera.name}"
    
    if sys.platform == 'darwin':
        cmd = [
            'ffplay',
            '-f', 'avfoundation',
            '-framerate', '30',
            '-video_size', '1280x720',
            '-i', camera.path,
            '-window_title', window_title,
            '-loglevel', 'error',
        ]
    else:
        cmd = [
            'ffplay',
            '-f', 'v4l2',
            '-framerate', '30',
            '-video_size', '1280x720',
            '-i', camera.path,
            '-window_title', window_title,
            '-loglevel', 'error',
        ]
    
    print(f"  {camera.name}: ffplay {camera.path}")
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def run_ffplay_mode(cameras: List[CameraInfo]):
    """Run in ffplay-only mode (no QR detection)."""
    print("\nLaunching ffplay windows...")
    print("  (QR detection disabled in ffplay mode)\n")
    
    processes = []
    for cam in cameras:
        proc = launch_ffplay(cam)
        processes.append(proc)
    
    print("\nPress Ctrl+C to stop, or close all windows.\n")
    
    def signal_handler(sig, frame):
        print("\nShutting down...")
        for proc in processes:
            proc.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Wait for processes
    try:
        while True:
            if all(proc.poll() is not None for proc in processes):
                print("All windows closed.")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    
    for proc in processes:
        proc.terminate()


def run_opencv_mode(cameras: List[CameraInfo]):
    """Run with OpenCV windows and QR detection."""
    print("\nOpening cameras with QR detection...")
    
    detector = QRDetector()
    caps: Dict[int, cv2.VideoCapture] = {}
    qr_status: Dict[int, Tuple[Optional[str], float]] = {}  # cam_idx -> (qr_data, timestamp)
    
    # Open all cameras
    for cam in cameras:
        if sys.platform == 'darwin':
            cap = cv2.VideoCapture(cam.index)
        else:
            cap = cv2.VideoCapture(cam.path)
        
        if cap.isOpened():
            # Try to set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            caps[cam.index] = cap
            qr_status[cam.index] = (None, 0)
            print(f"  Opened: {cam.name} ({cam.path})")
        else:
            print(f"  Failed to open: {cam.name} ({cam.path})")
    
    if not caps:
        print("\nNo cameras could be opened!")
        return
    
    print(f"\n{len(caps)} camera(s) active. Press 'q' to quit.\n")
    print("=" * 50)
    print("  QR Detection Status")
    print("=" * 50)
    
    running = True
    
    def signal_handler(sig, frame):
        nonlocal running
        running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    last_status: Dict[int, Optional[str]] = {}
    
    while running:
        for cam in cameras:
            if cam.index not in caps:
                continue
            
            cap = caps[cam.index]
            ret, frame = cap.read()
            
            if not ret or frame is None:
                continue
            
            # Detect QR
            qr_data = detector.detect(frame)
            now = time.time()
            qr_status[cam.index] = (qr_data, now)
            
            # Print status changes
            if qr_data != last_status.get(cam.index):
                if qr_data:
                    print(f"\033[92m  [Cam {cam.index}] QR: {qr_data}\033[0m")
                elif last_status.get(cam.index):
                    print(f"\033[91m  [Cam {cam.index}] QR: Lost\033[0m")
                last_status[cam.index] = qr_data
            
            # Draw overlay on frame
            h, w = frame.shape[:2]
            
            # Camera label
            cv2.putText(frame, f"Camera {cam.index}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # QR status
            if qr_data:
                color = (0, 255, 0)  # Green
                text = f"QR: {qr_data}"
            else:
                color = (0, 0, 255)  # Red
                text = "QR: Not detected"
            
            cv2.putText(frame, text, (10, h - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Status indicator circle
            cv2.circle(frame, (w - 30, 30), 15, color, -1)
            
            # Show frame
            window_name = f"Camera {cam.index}: {cam.name}"
            cv2.imshow(window_name, frame)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            running = False
        
        # Check if all windows were closed
        if cv2.getWindowProperty(list(caps.keys())[0] if caps else 0, cv2.WND_PROP_VISIBLE) < 1:
            # Window was closed
            pass
    
    # Cleanup
    print("\nShutting down...")
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description='Sync Camera Setup - Live camera preview with QR detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  Default    OpenCV windows with QR detection overlay
  --ffplay   Use ffplay for display (more compatible, but no QR detection)

Examples:
  %(prog)s                    # OpenCV mode with QR detection
  %(prog)s --ffplay           # ffplay mode (no QR)
  %(prog)s --cameras 0,2      # Only use cameras 0 and 2
        """
    )
    parser.add_argument('--ffplay', action='store_true', 
                        help='Use ffplay for display (no QR detection)')
    parser.add_argument('--cameras', type=str, 
                        help='Comma-separated camera indices to use (e.g., "0,2")')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  SYNC CAMERA SETUP")
    print("  Live camera preview with QR detection")
    print("=" * 60)
    
    # Discover cameras
    print("\nDiscovering cameras...")
    cameras = discover_cameras()
    
    # Filter by user selection
    if args.cameras:
        selected = [int(x.strip()) for x in args.cameras.split(',')]
        cameras = [c for c in cameras if c.index in selected]
    
    if not cameras:
        print("\nNo cameras found!")
        print("Make sure cameras are connected and not in use.")
        sys.exit(1)
    
    print(f"\nFound {len(cameras)} camera(s):")
    for cam in cameras:
        print(f"  - [{cam.index}] {cam.name} ({cam.path})")
    
    # Run appropriate mode
    if args.ffplay:
        run_ffplay_mode(cameras)
    else:
        run_opencv_mode(cameras)


if __name__ == '__main__':
    main()
