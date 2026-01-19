#!/usr/bin/env python3
"""
Sync Camera Setup - Live camera preview with QR code detection.
Opens all USB cameras (filtering out atlas/user-facing cameras) using ffplay
and runs QR detection in parallel.

Use this to verify cameras can see and decode the QR beacon before recording.
"""

import argparse
import subprocess
import sys
import re
import time
import threading
import signal
from pathlib import Path
from typing import Optional, List, Dict
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


def discover_cameras_linux() -> List[CameraInfo]:
    """Discover cameras on Linux using v4l2."""
    cameras = []
    try:
        video_devices = sorted(Path('/dev').glob('video*'))
        for device in video_devices:
            try:
                idx = int(device.name.replace('video', ''))
            except ValueError:
                continue
            
            name = f'Camera {idx}'
            
            # Try to get device name using v4l2-ctl
            try:
                result = subprocess.run(
                    ['v4l2-ctl', '-d', str(device), '--info'],
                    capture_output=True, text=True, timeout=5
                )
                for line in result.stdout.split('\n'):
                    if 'Card type' in line:
                        name = line.split(':', 1)[1].strip()
                        break
            except Exception:
                pass
            
            # Check if it's a capture device (not metadata/output)
            try:
                result = subprocess.run(
                    ['v4l2-ctl', '-d', str(device), '--list-formats-ext'],
                    capture_output=True, text=True, timeout=5
                )
                # Skip if no video capture formats
                if 'Video Capture' not in result.stdout and 'MJPG' not in result.stdout and 'YUYV' not in result.stdout:
                    continue
            except Exception:
                pass
            
            if not should_filter_camera(name):
                cameras.append(CameraInfo(
                    index=idx,
                    name=name,
                    path=str(device)
                ))
            else:
                print(f"  Filtered out: {name} ({device})")
    except Exception as e:
        print(f"Error discovering cameras: {e}")
    
    return cameras


def discover_cameras_macos() -> List[CameraInfo]:
    """Discover cameras on macOS."""
    cameras = []
    # On macOS, just probe indices
    for idx in range(10):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            # Try to get name via system_profiler
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


def launch_ffplay(camera: CameraInfo, window_title: str) -> subprocess.Popen:
    """Launch ffplay for a camera."""
    if sys.platform == 'darwin':
        # macOS uses avfoundation
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
        # Linux uses v4l2
        cmd = [
            'ffplay',
            '-f', 'v4l2',
            '-framerate', '30',
            '-video_size', '1280x720',
            '-i', camera.path,
            '-window_title', window_title,
            '-loglevel', 'error',
        ]
    
    print(f"  Launching: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def qr_detection_loop(camera: CameraInfo, detector: QRDetector, stop_event: threading.Event, status: Dict):
    """Run QR detection in background for a camera."""
    cap = cv2.VideoCapture(camera.path if sys.platform != 'darwin' else camera.index)
    
    if not cap.isOpened():
        print(f"  Warning: Could not open {camera.path} for QR detection")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    last_qr = None
    last_print_time = 0
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret and frame is not None:
            qr_data = detector.detect(frame)
            status[camera.index] = qr_data
            
            # Print when QR changes or periodically
            now = time.time()
            if qr_data != last_qr or (qr_data and now - last_print_time > 1.0):
                if qr_data:
                    print(f"\033[92m  [{camera.name}] QR: {qr_data}\033[0m")
                elif last_qr:
                    print(f"\033[91m  [{camera.name}] QR: Lost\033[0m")
                last_qr = qr_data
                last_print_time = now
        
        # Don't spin too fast
        time.sleep(0.05)
    
    cap.release()


def print_status_header():
    """Print the status header."""
    print("\n" + "=" * 60)
    print("  QR Detection Status (Ctrl+C to stop)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Sync Camera Setup - Live camera preview with QR detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool helps you set up cameras for QR-based sync calibration.

It will:
  1. Discover USB cameras (filtering out atlas, FaceTime, virtual cameras)
  2. Launch ffplay windows for each camera
  3. Run QR detection and print results to console

Usage:
  1. Start the QR beacon on another device/screen
  2. Run this tool to see live camera feeds
  3. Adjust cameras until QR codes are detected (green text)
  4. Start your recording when all cameras show QR values
        """
    )
    parser.add_argument('--no-qr', action='store_true', help='Disable QR detection (just show video)')
    parser.add_argument('--cameras', type=str, help='Comma-separated camera indices to use (e.g., "0,2")')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  SYNC CAMERA SETUP")
    print("  Live camera preview with QR detection")
    print("=" * 60)
    
    # Discover cameras
    print("\nDiscovering cameras...")
    cameras = discover_cameras()
    
    # Filter by user selection if provided
    if args.cameras:
        selected_indices = [int(x.strip()) for x in args.cameras.split(',')]
        cameras = [c for c in cameras if c.index in selected_indices]
    
    if not cameras:
        print("\nNo cameras found!")
        print("Make sure cameras are connected and not in use by another application.")
        sys.exit(1)
    
    print(f"\nFound {len(cameras)} camera(s):")
    for cam in cameras:
        print(f"  - {cam.name} ({cam.path})")
    
    # Launch ffplay for each camera
    print("\nLaunching video windows...")
    ffplay_processes = []
    for cam in cameras:
        title = f"Camera {cam.index}: {cam.name}"
        proc = launch_ffplay(cam, title)
        ffplay_processes.append(proc)
    
    # Setup QR detection threads
    stop_event = threading.Event()
    qr_threads = []
    qr_status: Dict[int, Optional[str]] = {}
    
    if not args.no_qr:
        print("\nStarting QR detection...")
        detector = QRDetector()
        
        for cam in cameras:
            thread = threading.Thread(
                target=qr_detection_loop,
                args=(cam, detector, stop_event, qr_status),
                daemon=True
            )
            thread.start()
            qr_threads.append(thread)
        
        print_status_header()
    else:
        print("\nQR detection disabled. Press Ctrl+C to stop.")
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\nShutting down...")
        stop_event.set()
        for proc in ffplay_processes:
            proc.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Wait for ffplay processes to exit
    try:
        while True:
            # Check if all ffplay processes have exited
            all_exited = all(proc.poll() is not None for proc in ffplay_processes)
            if all_exited:
                print("\nAll video windows closed.")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    
    # Cleanup
    stop_event.set()
    for proc in ffplay_processes:
        proc.terminate()
    
    print("Done.")


if __name__ == '__main__':
    main()
