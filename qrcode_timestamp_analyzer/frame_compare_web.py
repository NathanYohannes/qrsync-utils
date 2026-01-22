#!/usr/bin/env python3
"""
FastAPI-based side-by-side frame comparison viewer for two camera videos.
Allows dynamic video selection through the web interface with drag-and-drop support.
"""

import argparse
import sys
import base64
import os
import io
import json
import time
from pathlib import Path
from contextlib import asynccontextmanager
import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Optional
from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# Try to import matplotlib for graph generation
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, graphs will be disabled")

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


def _analyze_batch_worker(args):
    """
    Worker function for parallel analysis. Processes a batch of frames sequentially.
    Works with both threading and multiprocessing.
    """
    video1_path, video2_path, frame_numbers, worker_id, crop1, crop2 = args
    
    results = []
    
    def worker_apply_crop(frame, crop_config):
        """Apply crop to frame if configured."""
        if frame is None or crop_config is None:
            return frame
        x, y, w, h = crop_config['x'], crop_config['y'], crop_config['w'], crop_config['h']
        return frame[y:y+h, x:x+w]
    
    try:
        # Each worker opens its own video captures
        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)
        
        if not cap1.isOpened() or not cap2.isOpened():
            print(f"Worker {worker_id}: Failed to open video files")
            return results
        
        # Create WeChat detector for this worker
        try:
            wechat_detector = cv2.wechat_qrcode_WeChatQRCode()
        except Exception as e:
            print(f"Worker {worker_id}: WeChat detector failed: {e}")
            wechat_detector = None
        
        def decode_qr_fast(frame):
            if frame is None or wechat_detector is None:
                return None
            try:
                res, _ = wechat_detector.detectAndDecode(frame)
                if res and res[0]:
                    return res[0]
            except:
                pass
            return None
        
        # Sort frame numbers for sequential access (much faster than random seeks)
        sorted_frames = sorted(frame_numbers)
        
        # Process frames sequentially
        for frame_num in sorted_frames:
            # Seek and read frame 1
            cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret1, frame1 = cap1.read()
            pts_ms1 = cap1.get(cv2.CAP_PROP_POS_MSEC)
            pts1 = int(pts_ms1 * 1000) if pts_ms1 > 0 else None
            
            # Apply crop to frame 1
            frame1 = worker_apply_crop(frame1, crop1)
            
            # Seek and read frame 2
            cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret2, frame2 = cap2.read()
            pts_ms2 = cap2.get(cv2.CAP_PROP_POS_MSEC)
            pts2 = int(pts_ms2 * 1000) if pts_ms2 > 0 else None
            
            # Apply crop to frame 2
            frame2 = worker_apply_crop(frame2, crop2)
            
            if not ret1 or not ret2 or frame1 is None or frame2 is None:
                continue
            
            # Decode QR codes (fast mode - WeChat only)
            ts1 = decode_qr_fast(frame1)
            ts2 = decode_qr_fast(frame2)
            
            results.append((frame_num, ts1, ts2, pts1, pts2))
        
        cap1.release()
        cap2.release()
        
    except Exception as e:
        print(f"Worker {worker_id} error: {e}")
        import traceback
        traceback.print_exc()
    
    return results


class VideoLoadRequest(BaseModel):
    video1: str
    video2: str
    video3: Optional[str] = None
    video4: Optional[str] = None


# Stream type detection and crop configuration
STREAM_CONFIGS = {
    'stream_a': {
        'detect_width': 4000,  # 4000px wide = Stream A (combined)
        'crop': {'x': 2080, 'y': 0, 'w': 1920, 'h': None},  # Crop right portion
        'description': 'Stream A (4000px) - cropped to right 1920px'
    },
    'stream_b': {
        'detect_height': 480,  # 480px tall = Stream B (skip)
        'skip': True,
        'description': 'Stream B (480px tall) - skipped'
    }
}


def detect_stream_config(width: int, height: int) -> dict:
    """Detect stream type and return configuration.
    
    Stream detection rules:
    - Stream A: Width >= 4000px (combined multi-view) -> crop right portion (x=2080 to 4000)
    - Stream B: Width > 1000 AND height == 480px (specific combined stream type) -> skip
    - Normal: All other streams -> use as-is
    
    Note: Standard 640x480 videos are NOT considered Stream B.
    """
    # Check for Stream A (4000px+ wide combined stream)
    if width >= 4000:
        # Crop from x=2080 to end (keeping 1920px or whatever remains)
        crop_w = width - 2080
        return {
            'type': 'stream_a',
            'crop': {'x': 2080, 'y': 0, 'w': crop_w, 'h': height},
            'skip': False,
            'description': f'Stream A ({width}x{height}) -> cropped to {crop_w}x{height} (x=2080+)'
        }
    # Check for Stream B (wide combined stream that's 480px tall - NOT standard 640x480)
    # Only skip if it's clearly a combined/special stream (width > 1000)
    if width > 1000 and height == 480:
        return {
            'type': 'stream_b',
            'crop': None,
            'skip': True,
            'description': f'Stream B ({width}x{height}) - skipped (wide 480p stream)'
        }
    # Normal stream - no crop needed
    return {
        'type': 'normal',
        'crop': None,
        'skip': False,
        'description': f'Normal ({width}x{height})'
    }


def apply_crop(frame, crop_config: dict):
    """Apply crop to frame if configured."""
    if frame is None or crop_config is None:
        return frame
    x, y, w, h = crop_config['x'], crop_config['y'], crop_config['w'], crop_config['h']
    return frame[y:y+h, x:x+w]


class VideoFrameServer:
    def __init__(self, video1_path: str, video2_path: str, video3_path: str = None, video4_path: str = None):
        """Initialize with 2-4 video paths. Videos are auto-configured based on dimensions."""
        
        # Collect all provided video paths
        video_paths = [video1_path, video2_path]
        if video3_path:
            video_paths.append(video3_path)
        if video4_path:
            video_paths.append(video4_path)
        
        # Open all videos and detect stream types
        self.videos = []  # List of {cap, path, name, config, frame_count, fps, width, height}
        self.skipped_videos = []  # Videos that were skipped (Stream B)
        
        for vpath in video_paths:
            cap = cv2.VideoCapture(vpath)
            if not cap.isOpened():
                raise ValueError(f"Could not open {vpath}")
            
            raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            config = detect_stream_config(raw_width, raw_height)
            
            vname = Path(vpath).name
            print(f"  {vname}: {config['description']}")
            
            if config['skip']:
                cap.release()
                self.skipped_videos.append({'path': vpath, 'name': vname, 'config': config})
                continue
            
            # Calculate effective dimensions after crop
            if config['crop']:
                eff_width = config['crop']['w']
                eff_height = config['crop']['h']
            else:
                eff_width = raw_width
                eff_height = raw_height
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            self.videos.append({
                'cap': cap,
                'path': vpath,
                'name': vname,
                'config': config,
                'frame_count': frame_count,
                'fps': fps,
                'raw_width': raw_width,
                'raw_height': raw_height,
                'width': eff_width,
                'height': eff_height,
            })
        
        if len(self.videos) < 2:
            raise ValueError(f"Need at least 2 valid videos, got {len(self.videos)} (skipped {len(self.skipped_videos)})")
        
        # Fix frame counts for videos with unreliable metadata
        for v in self.videos:
            if v['frame_count'] <= 0 or v['frame_count'] > 10000000:
                cached = self._get_cached_frame_count(v['path'])
                if cached:
                    print(f"Using cached frame count for {v['name']}: {cached}")
                    v['frame_count'] = cached
                else:
                    print(f"Counting frames for {v['name']} (metadata unreliable)...")
                    v['frame_count'] = self._count_frames(v['path'])
                    self._save_frame_count_to_cache(v['path'], v['frame_count'])
                    # Reopen after counting
                    v['cap'].release()
                    v['cap'] = cv2.VideoCapture(v['path'])
        
        # Backward compatibility properties (for 2 videos)
        self.cap1 = self.videos[0]['cap']
        self.cap2 = self.videos[1]['cap']
        self.video1_path = self.videos[0]['path']
        self.video2_path = self.videos[1]['path']
        self.video1_name = self.videos[0]['name']
        self.video2_name = self.videos[1]['name']
        self.frame_count1 = self.videos[0]['frame_count']
        self.frame_count2 = self.videos[1]['frame_count']
        self.fps1 = self.videos[0]['fps']
        self.fps2 = self.videos[1]['fps']
        self.width1 = self.videos[0]['width']
        self.height1 = self.videos[0]['height']
        self.width2 = self.videos[1]['width']
        self.height2 = self.videos[1]['height']
        
        self.max_frames = min(v['frame_count'] for v in self.videos)
        self.num_cameras = len(self.videos)
        
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
    
    def _get_cached_frame_count(self, video_path: str) -> int:
        """Check for cached frame count in .frame_cache.json in the video's directory."""
        video_dir = Path(video_path).parent
        cache_file = video_dir / ".frame_cache.json"
        video_name = Path(video_path).name
        
        if cache_file.exists():
            try:
                import json
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
                if video_name in cache:
                    return cache[video_name]
            except (json.JSONDecodeError, IOError) as e:
                print(f"  Warning: Could not read cache file: {e}")
        return None
    
    def _save_frame_count_to_cache(self, video_path: str, frame_count: int):
        """Save frame count to .frame_cache.json in the video's directory."""
        video_dir = Path(video_path).parent
        cache_file = video_dir / ".frame_cache.json"
        video_name = Path(video_path).name
        
        cache = {}
        if cache_file.exists():
            try:
                import json
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        cache[video_name] = frame_count
        
        try:
            import json
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            print(f"  Saved frame count to cache: {cache_file}")
        except IOError as e:
            print(f"  Warning: Could not save to cache: {e}")
    
    def decode_qr(self, frame, fast_mode: bool = False) -> str:
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
        
        # Skip fallbacks in fast mode
        if fast_mode:
            return None
        
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
    
    def _seek_to_frame_sequential(self, video_idx: int, frame_num: int) -> tuple:
        """Reliable frame reading using sequential access with position tracking.
        
        For H.264 in MKV without proper indexes, this is the only reliable method.
        We track the current position and read sequentially.
        
        Returns: (frame, pts_us) tuple
        """
        video = self.videos[video_idx]
        
        # Track current position per video (next_frame = next frame number to be read)
        if 'next_frame' not in video:
            video['next_frame'] = 0
        
        next_frame = video['next_frame']
        
        # If we need to go backwards or jump far forward, reopen the video
        if frame_num < next_frame or frame_num > next_frame + 300:
            # Reopen video capture to ensure clean state
            video['cap'].release()
            video['cap'] = cv2.VideoCapture(video['path'])
            video['next_frame'] = 0
            next_frame = 0
        
        cap = video['cap']
        
        # Read frames until we've read the target frame
        frame = None
        pts_ms = None
        while video['next_frame'] <= frame_num:
            ret, frame = cap.read()
            if not ret:
                return None, None
            pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            video['next_frame'] += 1
        
        pts_us = int(pts_ms * 1000) if pts_ms and pts_ms > 0 else None
        return frame, pts_us
    
    def get_frame(self, cam: int, frame_num: int) -> tuple:
        """Get frame with base64 encoding. cam is 1-indexed."""
        cache_key = (cam, frame_num)
        
        if cache_key in self.frame_cache:
            return self.frame_cache[cache_key]
        
        cam_idx = cam - 1  # Convert to 0-indexed
        if cam_idx < 0 or cam_idx >= len(self.videos):
            return None, None, None
        
        video = self.videos[cam_idx]
        
        with video_lock:
            frame, pts_us = self._seek_to_frame_sequential(cam_idx, frame_num)
        
        if frame is None:
            return None, None, None
        
        # Apply crop if configured
        frame = apply_crop(frame, video['config'].get('crop'))
        
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
        info = {
            # Backward compatible fields for 2 videos
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
            # New multi-video fields
            'num_cameras': self.num_cameras,
            'videos': [
                {
                    'name': v['name'],
                    'path': v['path'],
                    'frame_count': v['frame_count'],
                    'fps': v['fps'],
                    'width': v['width'],
                    'height': v['height'],
                    'raw_width': v['raw_width'],
                    'raw_height': v['raw_height'],
                    'stream_type': v['config']['type'],
                    'description': v['config']['description'],
                }
                for v in self.videos
            ],
            'skipped_videos': [
                {
                    'name': v['name'],
                    'path': v['path'],
                    'reason': v['config']['description'],
                }
                for v in self.skipped_videos
            ],
        }
        return info
    
    def get_frame_raw(self, cam: int, frame_num: int, fast_mode: bool = False) -> tuple:
        """Get raw frame data without base64 encoding. cam is 1-indexed."""
        cam_idx = cam - 1  # Convert to 0-indexed
        if cam_idx < 0 or cam_idx >= len(self.videos):
            return None, None, None
        
        video = self.videos[cam_idx]
        
        with video_lock:
            frame, pts_us = self._seek_to_frame_sequential(cam_idx, frame_num)
        
        if frame is None:
            return None, None, None
        
        # Apply crop if configured
        frame = apply_crop(frame, video['config'].get('crop'))
        
        timestamp = self.decode_qr(frame, fast_mode=fast_mode)
        return frame, timestamp, pts_us
    
    def analyze_sync(self, sample_interval: int = 30, progress_callback=None, fast_mode: bool = True) -> dict:
        """
        Analyze sync across the entire video by sampling frames.
        
        Args:
            sample_interval: Sample every N frames (default: 30 = 1 per second at 30fps)
            progress_callback: Optional callback(current, total) for progress updates
            fast_mode: If True, only use WeChat QR detector (faster). If False, try all detectors.
        
        Returns:
            Dictionary with analysis results
        """
        results = {
            'frames': [],
            'qr_offsets': [],
            'pts_offsets': [],
            'sync_errors': [],
            'qr_detections': {'cam1': 0, 'cam2': 0, 'both': 0},
            'sample_count': 0,
            'video1_name': self.video1_name,
            'video2_name': self.video2_name,
            'max_frames': self.max_frames,
            'fps': self.fps1,
            'sample_interval': sample_interval,
        }
        
        total_samples = self.max_frames // sample_interval
        
        for i, frame_num in enumerate(range(0, self.max_frames, sample_interval)):
            if progress_callback:
                progress_callback(i, total_samples)
            
            # Get data for both cameras
            frame1, ts1, pts1 = self.get_frame_raw(1, frame_num, fast_mode=fast_mode)
            frame2, ts2, pts2 = self.get_frame_raw(2, frame_num, fast_mode=fast_mode)
            
            if frame1 is None or frame2 is None:
                continue
            
            results['sample_count'] += 1
            results['frames'].append(frame_num)
            
            # Track QR detection success
            if ts1:
                results['qr_detections']['cam1'] += 1
            if ts2:
                results['qr_detections']['cam2'] += 1
            if ts1 and ts2:
                results['qr_detections']['both'] += 1
            
            # Calculate offsets
            qr_offset = None
            pts_offset = None
            sync_error = None
            
            if ts1 and ts2:
                try:
                    qr_offset = (int(ts2) - int(ts1)) / 1000.0  # Convert us to ms
                except ValueError:
                    pass
            
            if pts1 and pts2:
                pts_offset = (pts2 - pts1) / 1000.0  # Convert us to ms
            
            if qr_offset is not None and pts_offset is not None:
                sync_error = pts_offset - qr_offset
            
            results['qr_offsets'].append(qr_offset)
            results['pts_offsets'].append(pts_offset)
            results['sync_errors'].append(sync_error)
        
        # Calculate statistics
        results['statistics'] = self._calculate_statistics(results)
        
        return results
    
    def analyze_sync_fast(self, sample_interval: int = 30, progress_callback=None, num_workers: int = None) -> dict:
        """
        Fast parallel analysis using multiprocessing and sequential batch reads.
        
        Args:
            sample_interval: Sample every N frames (default: 30 = 1 per second at 30fps)
            progress_callback: Optional callback(current, total) for progress updates
            num_workers: Number of parallel workers (default: CPU count)
        
        Returns:
            Dictionary with analysis results
        """
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 8)  # Cap at 8 workers
        
        results = {
            'frames': [],
            'qr_offsets': [],
            'pts_offsets': [],
            'sync_errors': [],
            'qr_detections': {'cam1': 0, 'cam2': 0, 'both': 0},
            'sample_count': 0,
            'video1_name': self.video1_name,
            'video2_name': self.video2_name,
            'max_frames': self.max_frames,
            'fps': self.fps1,
            'sample_interval': sample_interval,
        }
        
        # Generate list of frame numbers to analyze
        frame_numbers = list(range(0, self.max_frames, sample_interval))
        total_samples = len(frame_numbers)
        
        if total_samples == 0:
            return results
        
        # Split frames into batches for workers
        # Each worker gets a contiguous range for optimal sequential reading
        batch_size = max(1, total_samples // num_workers)
        batches = []
        
        for i in range(0, total_samples, batch_size):
            batch_frames = frame_numbers[i:i + batch_size]
            if batch_frames:
                batches.append((
                    self.video1_path,
                    self.video2_path,
                    batch_frames,
                    len(batches)  # worker_id
                ))
        
        if progress_callback:
            progress_callback(0, total_samples)
        
        # Process batches in parallel using threads (works better with OpenCV on macOS)
        all_results = []
        completed = 0
        
        print(f"Starting analysis with {num_workers} threads, {len(batches)} batches, {total_samples} total samples")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_analyze_batch_worker, batch): i for i, batch in enumerate(batches)}
            
            for future in as_completed(futures):
                try:
                    batch_results = future.result(timeout=3600)  # 1 hour timeout per batch
                    all_results.extend(batch_results)
                    completed += len(batch_results)
                    
                    if progress_callback:
                        progress_callback(completed, total_samples)
                    
                    print(f"Batch completed: {completed}/{total_samples} samples ({100*completed//total_samples}%)")
                except Exception as e:
                    print(f"Worker error: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Sort results by frame number (workers may complete out of order)
        all_results.sort(key=lambda x: x[0])
        
        # Process results
        for frame_num, ts1, ts2, pts1, pts2 in all_results:
            results['sample_count'] += 1
            results['frames'].append(frame_num)
            
            # Track QR detection success
            if ts1:
                results['qr_detections']['cam1'] += 1
            if ts2:
                results['qr_detections']['cam2'] += 1
            if ts1 and ts2:
                results['qr_detections']['both'] += 1
            
            # Calculate offsets
            qr_offset = None
            pts_offset = None
            sync_error = None
            
            if ts1 and ts2:
                try:
                    qr_offset = (int(ts2) - int(ts1)) / 1000.0  # Convert us to ms
                except ValueError:
                    pass
            
            if pts1 and pts2:
                pts_offset = (pts2 - pts1) / 1000.0  # Convert us to ms
            
            if qr_offset is not None and pts_offset is not None:
                sync_error = pts_offset - qr_offset
            
            results['qr_offsets'].append(qr_offset)
            results['pts_offsets'].append(pts_offset)
            results['sync_errors'].append(sync_error)
        
        # Calculate statistics
        results['statistics'] = self._calculate_statistics(results)
        
        return results
    
    def _calculate_statistics(self, results: dict) -> dict:
        """Calculate statistics from analysis results."""
        stats = {}
        
        for key in ['qr_offsets', 'pts_offsets', 'sync_errors']:
            values = [v for v in results[key] if v is not None]
            if values:
                stats[key] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'p5': np.percentile(values, 5),
                    'p95': np.percentile(values, 95),
                }
            else:
                stats[key] = None
        
        # QR detection rate
        total = results['sample_count']
        if total > 0:
            stats['qr_detection_rate'] = {
                'cam1': results['qr_detections']['cam1'] / total * 100,
                'cam2': results['qr_detections']['cam2'] / total * 100,
                'both': results['qr_detections']['both'] / total * 100,
            }
        
        return stats
    
    def generate_sync_graph(self, results: dict) -> bytes:
        """Generate a sync graph image from analysis results."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), dpi=100)
        fig.patch.set_facecolor('#0a0a0f')
        
        frames = results['frames']
        time_seconds = [f / results['fps'] for f in frames]
        time_minutes = [t / 60 for t in time_seconds]
        
        # Style settings
        for ax in axes:
            ax.set_facecolor('#12121a')
            ax.tick_params(colors='#8888a0')
            ax.spines['bottom'].set_color('#2a2a3a')
            ax.spines['top'].set_color('#2a2a3a')
            ax.spines['left'].set_color('#2a2a3a')
            ax.spines['right'].set_color('#2a2a3a')
            ax.xaxis.label.set_color('#e8e8ed')
            ax.yaxis.label.set_color('#e8e8ed')
            ax.title.set_color('#e8e8ed')
        
        # Plot 1: QR Offset over time
        qr_data = [(t, v) for t, v in zip(time_minutes, results['qr_offsets']) if v is not None]
        if qr_data:
            t_qr, v_qr = zip(*qr_data)
            axes[0].plot(t_qr, v_qr, color='#00ff88', linewidth=1, alpha=0.8, label='QR Offset')
            axes[0].axhline(y=0, color='#ffffff', linestyle='--', alpha=0.3)
            stats = results['statistics'].get('qr_offsets')
            if stats:
                axes[0].axhline(y=stats['mean'], color='#00d4ff', linestyle='-', alpha=0.5, label=f"Mean: {stats['mean']:.2f}ms")
                axes[0].fill_between(t_qr, stats['mean'] - stats['std'], stats['mean'] + stats['std'], 
                                     color='#00d4ff', alpha=0.1)
        axes[0].set_ylabel('QR Offset (ms)', fontsize=10)
        axes[0].set_title('QR Timestamp Offset (Cam2 - Cam1)', fontsize=12, fontweight='bold')
        axes[0].legend(loc='upper right', facecolor='#1a1a25', edgecolor='#2a2a3a', labelcolor='#e8e8ed')
        axes[0].grid(True, alpha=0.2, color='#2a2a3a')
        
        # Plot 2: PTS Offset over time
        pts_data = [(t, v) for t, v in zip(time_minutes, results['pts_offsets']) if v is not None]
        if pts_data:
            t_pts, v_pts = zip(*pts_data)
            axes[1].plot(t_pts, v_pts, color='#00aaff', linewidth=1, alpha=0.8, label='PTS Offset')
            axes[1].axhline(y=0, color='#ffffff', linestyle='--', alpha=0.3)
            stats = results['statistics'].get('pts_offsets')
            if stats:
                axes[1].axhline(y=stats['mean'], color='#ff6688', linestyle='-', alpha=0.5, label=f"Mean: {stats['mean']:.2f}ms")
        axes[1].set_ylabel('PTS Offset (ms)', fontsize=10)
        axes[1].set_title('PTS Offset (Cam2 - Cam1)', fontsize=12, fontweight='bold')
        axes[1].legend(loc='upper right', facecolor='#1a1a25', edgecolor='#2a2a3a', labelcolor='#e8e8ed')
        axes[1].grid(True, alpha=0.2, color='#2a2a3a')
        
        # Plot 3: Sync Error over time
        sync_data = [(t, v) for t, v in zip(time_minutes, results['sync_errors']) if v is not None]
        if sync_data:
            t_sync, v_sync = zip(*sync_data)
            # Color code by magnitude
            colors = ['#00ff88' if abs(v) < 5 else '#ffaa00' if abs(v) < 20 else '#ff4466' for v in v_sync]
            axes[2].scatter(t_sync, v_sync, c=colors, s=10, alpha=0.7)
            axes[2].axhline(y=0, color='#ffffff', linestyle='--', alpha=0.3)
            axes[2].axhline(y=5, color='#ffaa00', linestyle=':', alpha=0.3)
            axes[2].axhline(y=-5, color='#ffaa00', linestyle=':', alpha=0.3)
            stats = results['statistics'].get('sync_errors')
            if stats:
                axes[2].axhline(y=stats['mean'], color='#ff6688', linestyle='-', alpha=0.5, label=f"Mean: {stats['mean']:.2f}ms")
        axes[2].set_xlabel('Time (minutes)', fontsize=10)
        axes[2].set_ylabel('Sync Error (ms)', fontsize=10)
        axes[2].set_title('Sync Error (PTS - QR) - Shows PTS Accuracy', fontsize=12, fontweight='bold')
        axes[2].legend(loc='upper right', facecolor='#1a1a25', edgecolor='#2a2a3a', labelcolor='#e8e8ed')
        axes[2].grid(True, alpha=0.2, color='#2a2a3a')
        
        plt.tight_layout()
        
        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='#0a0a0f', edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    
    def generate_analysis_video(self, output_path: str, sample_interval: int = 1, 
                                 progress_callback=None, max_frames: int = None,
                                 output_fps: float = 30.0, only_with_qr: bool = True) -> str:
        """
        Generate a side-by-side analysis video with overlaid sync data.
        
        Args:
            output_path: Path to save the output video
            sample_interval: Process every N frames (1 = all frames)
            progress_callback: Optional callback(current, total, status) for progress
            max_frames: Maximum frames to process (None = all)
            output_fps: FPS of output video (default 30 for smooth playback)
            only_with_qr: If True, only include frames where both cameras have QR data
        
        Returns:
            Path to generated video
        """
        # Determine output dimensions
        out_width = self.width1 + self.width2
        out_height = max(self.height1, self.height2) + 80  # Extra height for overlay
        
        # Setup video writer - always output at smooth fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, output_fps, (out_width, out_height))
        
        if not writer.isOpened():
            raise ValueError(f"Could not open video writer for {output_path}")
        
        total = min(max_frames or self.max_frames, self.max_frames)
        frames_written = 0
        frames_skipped = 0
        
        try:
            for frame_num in range(0, total, sample_interval):
                if progress_callback:
                    status = f"Processing... ({frames_written} frames written, {frames_skipped} skipped)"
                    progress_callback(frame_num, total, status)
                
                # Get frames
                frame1, ts1, pts1 = self.get_frame_raw(1, frame_num, fast_mode=True)
                frame2, ts2, pts2 = self.get_frame_raw(2, frame_num, fast_mode=True)
                
                if frame1 is None or frame2 is None:
                    frames_skipped += 1
                    continue
                
                # Calculate offsets
                qr_offset = None
                pts_offset = None
                sync_error = None
                
                if ts1 and ts2:
                    try:
                        qr_offset = (int(ts2) - int(ts1)) / 1000.0
                    except ValueError:
                        pass
                
                if pts1 and pts2:
                    pts_offset = (pts2 - pts1) / 1000.0
                
                if qr_offset is not None and pts_offset is not None:
                    sync_error = pts_offset - qr_offset
                
                # Skip frames without QR data in both cameras if only_with_qr is True
                if only_with_qr and qr_offset is None:
                    frames_skipped += 1
                    continue
                
                # Create combined frame
                combined = np.zeros((out_height, out_width, 3), dtype=np.uint8)
                combined[:] = (15, 18, 26)  # Dark background #12121a
                
                # Place frames
                y_offset = 80
                combined[y_offset:y_offset + self.height1, 0:self.width1] = frame1
                combined[y_offset:y_offset + self.height2, self.width1:self.width1 + self.width2] = frame2
                
                # Draw overlay
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                # Frame number
                cv2.putText(combined, f"Frame: {frame_num:,}", (10, 25), 
                           font, 0.6, (0, 212, 255), 1, cv2.LINE_AA)
                
                # Time in source video
                time_sec = frame_num / self.fps1
                time_str = f"{int(time_sec // 3600):02d}:{int((time_sec % 3600) // 60):02d}:{time_sec % 60:05.2f}"
                cv2.putText(combined, time_str, (200, 25), 
                           font, 0.6, (136, 136, 160), 1, cv2.LINE_AA)
                
                # QR timestamps
                ts1_str = f"QR1: {ts1}" if ts1 else "QR1: —"
                ts2_str = f"QR2: {ts2}" if ts2 else "QR2: —"
                cv2.putText(combined, ts1_str, (10, 50), font, 0.5, (0, 255, 136), 1, cv2.LINE_AA)
                cv2.putText(combined, ts2_str, (self.width1 + 10, 50), font, 0.5, (0, 255, 136), 1, cv2.LINE_AA)
                
                # Offsets
                if qr_offset is not None:
                    color = (0, 255, 136) if abs(qr_offset) < 20 else (0, 170, 255) if abs(qr_offset) < 50 else (68, 102, 255)
                    cv2.putText(combined, f"QR Offset: {qr_offset:+.1f}ms", (400, 25), 
                               font, 0.6, color, 1, cv2.LINE_AA)
                
                if sync_error is not None:
                    color = (0, 255, 136) if abs(sync_error) < 5 else (0, 170, 255) if abs(sync_error) < 20 else (68, 102, 255)
                    cv2.putText(combined, f"Sync Error: {sync_error:+.1f}ms", (600, 25), 
                               font, 0.6, color, 1, cv2.LINE_AA)
                
                # Camera labels
                cv2.putText(combined, "CAM1", (10, 70), font, 0.5, (0, 170, 255), 1, cv2.LINE_AA)
                cv2.putText(combined, "CAM2", (self.width1 + 10, 70), font, 0.5, (255, 102, 136), 1, cv2.LINE_AA)
                
                writer.write(combined)
                frames_written += 1
        
        finally:
            writer.release()
        
        if progress_callback:
            progress_callback(total, total, f"Complete! {frames_written} frames written, {frames_skipped} skipped")
        
        return output_path
    
    def close(self):
        for v in self.videos:
            v['cap'].release()


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
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .drop-zones-4 {
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
        }
        
        .cam3 { --cam-color: #aa66ff; }
        .cam4 { --cam-color: #ffaa00; }
        .drop-zone.cam3 { border-color: var(--cam-color); }
        .drop-zone.cam4 { border-color: var(--cam-color); }
        .drop-zone.cam3 .drop-zone-label { color: var(--cam-color); }
        .drop-zone.cam4 .drop-zone-label { color: var(--cam-color); }
        .path-input-group.cam3 label { color: #aa66ff; }
        .path-input-group.cam4 label { color: #ffaa00; }
        
        .optional-label {
            font-size: 0.7rem;
            color: var(--text-secondary);
            margin-top: 4px;
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
        
        .video-option.cam3 {
            border-color: #aa66ff;
            color: #aa66ff;
            background: rgba(170, 102, 255, 0.1);
        }
        
        .video-option.cam4 {
            border-color: #ffaa00;
            color: #ffaa00;
            background: rgba(255, 170, 0, 0.1);
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
        
        <div class="drop-zones drop-zones-4">
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
            
            <div class="drop-zone cam3" id="dropzone3" onclick="document.getElementById('file3').click()">
                <div class="drop-zone-label">📷 Camera 3</div>
                <div class="drop-zone-icon">📂</div>
                <div class="drop-zone-text">Drop video here</div>
                <div class="drop-zone-or">— or —</div>
                <button class="browse-btn" onclick="event.stopPropagation(); document.getElementById('file3').click()">Browse Files</button>
                <input type="file" id="file3" accept=".mkv,.mp4,.avi,.mov" onchange="handleFileSelect(3, this)">
                <div class="selected-file" id="selected3" style="display: none;"></div>
                <div class="optional-label">(optional)</div>
            </div>
            
            <div class="drop-zone cam4" id="dropzone4" onclick="document.getElementById('file4').click()">
                <div class="drop-zone-label">📷 Camera 4</div>
                <div class="drop-zone-icon">📂</div>
                <div class="drop-zone-text">Drop video here</div>
                <div class="drop-zone-or">— or —</div>
                <button class="browse-btn" onclick="event.stopPropagation(); document.getElementById('file4').click()">Browse Files</button>
                <input type="file" id="file4" accept=".mkv,.mp4,.avi,.mov" onchange="handleFileSelect(4, this)">
                <div class="selected-file" id="selected4" style="display: none;"></div>
                <div class="optional-label">(optional)</div>
            </div>
        </div>
        
        <button class="btn btn-primary" id="compare-btn" onclick="loadVideos()" disabled>
            Select At Least 2 Videos to Compare
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
            <div class="path-input-group cam3">
                <label>Camera 3 Full Path (optional)</label>
                <input type="text" id="video3-path" placeholder="/full/path/to/camera3.mkv">
            </div>
            <div class="path-input-group cam4">
                <label>Camera 4 Full Path (optional)</label>
                <input type="text" id="video4-path" placeholder="/full/path/to/camera4.mkv">
            </div>
        </div>
        
        <p class="instructions">
            💡 <strong>Tip:</strong> First scan a directory, then drop or browse for videos.<br>
            The app will automatically find the full path for dropped files.
        </p>
    </div>
    
    <script>
        let selectedFiles = { 1: null, 2: null, 3: null, 4: null };
        let resolvedPaths = { 1: null, 2: null, 3: null, 4: null };
        
        // Initialize
        document.getElementById('scan-dir').value = 'DEFAULT_DIR';
        
        // Setup drag and drop for all 4 drop zones
        ['dropzone1', 'dropzone2', 'dropzone3', 'dropzone4'].forEach((id, idx) => {
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
            const paths = [1, 2, 3, 4].map(n => 
                resolvedPaths[n] || document.getElementById('video' + n + '-path').value.trim()
            );
            
            const selectedCount = paths.filter(p => p).length;
            
            if (selectedCount >= 2) {
                btn.disabled = false;
                btn.textContent = `Compare ${selectedCount} Videos →`;
            } else {
                btn.disabled = true;
                const missing = [];
                if (!paths[0]) missing.push('Camera 1');
                if (!paths[1]) missing.push('Camera 2');
                btn.textContent = 'Select at least ' + missing.join(' and ');
            }
        }
        
        async function loadVideos() {
            const path1 = resolvedPaths[1] || document.getElementById('video1-path').value.trim();
            const path2 = resolvedPaths[2] || document.getElementById('video2-path').value.trim();
            const path3 = resolvedPaths[3] || document.getElementById('video3-path').value.trim();
            const path4 = resolvedPaths[4] || document.getElementById('video4-path').value.trim();
            const errorEl = document.getElementById('error');
            
            if (!path1 || !path2) {
                errorEl.textContent = 'Please select at least Camera 1 and Camera 2';
                errorEl.style.display = 'block';
                return;
            }
            
            errorEl.style.display = 'none';
            
            try {
                const payload = { video1: path1, video2: path2 };
                if (path3) payload.video3 = path3;
                if (path4) payload.video4 = path4;
                
                const resp = await fetch('/api/load', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
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
            // Toggle selection: clicks cycle through cam1-4, then deselect
            // First check if already selected somewhere
            for (let cam = 1; cam <= 4; cam++) {
                if (resolvedPaths[cam] === path) {
                    // Deselect
                    resolvedPaths[cam] = null;
                    document.getElementById('video' + cam + '-path').value = '';
                    document.getElementById('selected' + cam).style.display = 'none';
                    document.getElementById('dropzone' + cam).classList.remove('has-file');
                    updateCompareButton();
                    refreshGridSelection();
                    return;
                }
            }
            
            // Find first empty slot
            for (let cam = 1; cam <= 4; cam++) {
                if (!resolvedPaths[cam]) {
                    resolvedPaths[cam] = path;
                    document.getElementById('video' + cam + '-path').value = path;
                    const el = document.getElementById('selected' + cam);
                    el.textContent = '✓ ' + path.split('/').pop();
                    el.className = 'selected-file found';
                    el.style.display = 'block';
                    document.getElementById('dropzone' + cam).classList.add('has-file');
                    break;
                }
            }
            
            updateCompareButton();
            refreshGridSelection();
        }
        
        function refreshGridSelection() {
            document.querySelectorAll('.video-option').forEach(el => {
                const path = el.getAttribute('title');
                el.classList.remove('cam1', 'cam2', 'cam3', 'cam4');
                if (resolvedPaths[1] === path) el.classList.add('cam1');
                if (resolvedPaths[2] === path) el.classList.add('cam2');
                if (resolvedPaths[3] === path) el.classList.add('cam3');
                if (resolvedPaths[4] === path) el.classList.add('cam4');
            });
        }
        
        function toggleAdvanced() {
            const el = document.getElementById('path-inputs');
            el.classList.toggle('show');
        }
        
        // Listen for manual path changes on all 4 inputs
        [1, 2, 3, 4].forEach(cam => {
            document.getElementById('video' + cam + '-path').addEventListener('input', () => {
                resolvedPaths[cam] = document.getElementById('video' + cam + '-path').value.trim() || null;
                updateCompareButton();
            });
            document.getElementById('video' + cam + '-path').addEventListener('keypress', e => { 
                if (e.key === 'Enter') loadVideos(); 
            });
        });
        
        // Enter key support for scan
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
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .video-container.cameras-3,
        .video-container.cameras-4 {
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
        }
        
        .video-panel {
            background: var(--bg-secondary);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border);
        }
        
        .video-panel.hidden {
            display: none;
        }
        
        .video-panel.cam3 { border-color: #aa66ff; }
        .video-panel.cam4 { border-color: #ffaa00; }
        
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
            
            <div class="video-panel cam3 hidden" id="panel3">
                <div class="video-header" style="flex-wrap: wrap; gap: 8px;">
                    <div class="video-title" style="color: #aa66ff;">📷 Cam3: <span id="video3-name">—</span></div>
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <div class="timestamp none" id="pts3" style="background: rgba(170, 102, 255, 0.2); color: #aa66ff;">PTS: —</div>
                        <div class="timestamp none" id="ts3">QR: —</div>
                    </div>
                </div>
                <div class="frame-wrapper">
                    <div class="loading" id="loading3">Loading...</div>
                    <img id="frame3" alt="Camera 3">
                </div>
            </div>
            
            <div class="video-panel cam4 hidden" id="panel4">
                <div class="video-header" style="flex-wrap: wrap; gap: 8px;">
                    <div class="video-title" style="color: #ffaa00;">📷 Cam4: <span id="video4-name">—</span></div>
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <div class="timestamp none" id="pts4" style="background: rgba(255, 170, 0, 0.2); color: #ffaa00;">PTS: —</div>
                        <div class="timestamp none" id="ts4">QR: —</div>
                    </div>
                </div>
                <div class="frame-wrapper">
                    <div class="loading" id="loading4">Loading...</div>
                    <img id="frame4" alt="Camera 4">
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
        
        <!-- Analysis Section -->
        <div class="analysis-section" style="margin-top: 20px; background: var(--bg-secondary); border-radius: 12px; padding: 20px; border: 1px solid var(--border);">
            <h3 style="color: var(--accent); margin-bottom: 15px; font-size: 1.1rem;">📊 Sync Analysis</h3>
            
            <div style="display: flex; gap: 15px; margin-bottom: 15px; flex-wrap: wrap; align-items: center;">
                <button class="btn" id="analyze-btn" onclick="runAnalysis()" style="background: linear-gradient(135deg, #00aa55, #00cc66); border: none; color: #fff; font-weight: 600; padding: 12px 24px;">
                    📈 Generate Analysis & Graph
                </button>
                <button class="btn" id="video-btn" onclick="generateVideo()" style="background: linear-gradient(135deg, #aa5500, #cc6600); border: none; color: #fff; font-weight: 600; padding: 12px 24px;">
                    🎬 Generate Analysis Video
                </button>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <label style="color: var(--text-secondary); font-size: 0.85rem;">Sample interval:</label>
                    <input type="number" id="sample-interval" value="30" min="1" max="1000" style="width: 80px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; padding: 8px; border: 1px solid var(--border); background: var(--bg-tertiary); color: var(--text-primary); border-radius: 6px;">
                </div>
                <label style="display: flex; align-items: center; gap: 6px; cursor: pointer; color: var(--text-secondary); font-size: 0.85rem;">
                    <input type="checkbox" id="fast-mode" checked style="width: 18px; height: 18px; accent-color: var(--accent);">
                    <span>⚡ Fast mode</span>
                    <span style="font-size: 0.75rem; color: var(--text-secondary);">(multiprocessing)</span>
                </label>
            </div>
            
            <div id="analysis-status" style="display: none; padding: 15px; background: var(--bg-tertiary); border-radius: 8px; margin-bottom: 15px; border: 1px solid var(--border);">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <div class="spinner" style="width: 18px; height: 18px; border: 2px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 1s linear infinite;"></div>
                        <span id="analysis-status-text" style="font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; color: var(--text-primary);">Running...</span>
                    </div>
                    <span id="analysis-percent" style="font-family: 'JetBrains Mono', monospace; font-size: 1.2rem; font-weight: 600; color: var(--accent);">0%</span>
                </div>
                <div style="background: var(--bg-primary); border-radius: 6px; height: 12px; overflow: hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,0.3);">
                    <div id="analysis-progress" style="height: 100%; background: linear-gradient(90deg, var(--accent), #00ff88); width: 0%; transition: width 0.2s ease; box-shadow: 0 0 10px rgba(0, 212, 255, 0.5);"></div>
                </div>
            </div>
            
            <div id="analysis-results" style="display: none;">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px;">
                    <div class="stat-card" style="background: var(--bg-tertiary); padding: 15px; border-radius: 8px;">
                        <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 5px;">QR Offset Mean</div>
                        <div id="stat-qr-mean" style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; color: var(--success);">—</div>
                    </div>
                    <div class="stat-card" style="background: var(--bg-tertiary); padding: 15px; border-radius: 8px;">
                        <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 5px;">QR Offset Std Dev</div>
                        <div id="stat-qr-std" style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; color: var(--warning);">—</div>
                    </div>
                    <div class="stat-card" style="background: var(--bg-tertiary); padding: 15px; border-radius: 8px;">
                        <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 5px;">Sync Error Mean</div>
                        <div id="stat-sync-mean" style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; color: var(--accent);">—</div>
                    </div>
                    <div class="stat-card" style="background: var(--bg-tertiary); padding: 15px; border-radius: 8px;">
                        <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 5px;">QR Detection Rate</div>
                        <div id="stat-qr-rate" style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; color: var(--success);">—</div>
                    </div>
                    <div class="stat-card" style="background: var(--bg-tertiary); padding: 15px; border-radius: 8px;">
                        <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 5px;">Samples Analyzed</div>
                        <div id="stat-samples" style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; color: var(--text-primary);">—</div>
                    </div>
                    <div class="stat-card" style="background: var(--bg-tertiary); padding: 15px; border-radius: 8px;">
                        <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 5px;">QR Range (min/max)</div>
                        <div id="stat-qr-range" style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; color: var(--text-primary);">—</div>
                    </div>
                </div>
                
                <div style="display: flex; gap: 10px; margin-bottom: 15px;">
                    <a href="/api/analyze/csv" class="btn" style="background: linear-gradient(135deg, #5555ff, #8888ff); border: none; color: #fff; font-weight: 600; padding: 10px 20px; text-decoration: none; display: inline-flex; align-items: center; gap: 8px;">
                        📥 Download CSV
                    </a>
                    <a href="/api/analyze/graph" target="_blank" class="btn" style="background: var(--bg-tertiary); border: 1px solid var(--border); color: var(--text-primary); padding: 10px 20px; text-decoration: none; display: inline-flex; align-items: center; gap: 8px;">
                        🖼️ Open Graph in New Tab
                    </a>
                </div>
                
                <div id="graph-container" style="background: var(--bg-primary); border-radius: 8px; padding: 10px; text-align: center;">
                    <img id="sync-graph" style="max-width: 100%; border-radius: 8px;" alt="Sync Analysis Graph">
                </div>
            </div>
            
            <div id="video-result" style="display: none; margin-top: 15px; padding: 15px; background: rgba(0, 255, 136, 0.1); border: 1px solid var(--success); border-radius: 8px;">
                <div style="color: var(--success); font-weight: 600; margin-bottom: 5px;">✓ Video Generated Successfully</div>
                <div id="video-path" style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: var(--text-secondary);"></div>
            </div>
        </div>
        
        <style>
            @keyframes spin { to { transform: rotate(360deg); } }
        </style>
    </div>
    
    <script>
        let currentFrame = 0, maxFrames = 0, fps = 30, isPlaying = false, playInterval = null, pendingRequest = null;
        let numCameras = 2;
        
        async function init() {
            try {
                const resp = await fetch('/api/info');
                if (!resp.ok) { window.location.href = '/'; return; }
                const info = await resp.json();
                
                maxFrames = info.max_frames;
                fps = info.fps1 || 30;
                numCameras = info.num_cameras || 2;
                
                // Show/hide camera panels based on num_cameras
                const container = document.querySelector('.video-container');
                if (numCameras >= 3) {
                    document.getElementById('panel3').classList.remove('hidden');
                    container.classList.add('cameras-' + numCameras);
                }
                if (numCameras >= 4) {
                    document.getElementById('panel4').classList.remove('hidden');
                }
                
                // Set video names for all cameras
                document.getElementById('video1-name').textContent = info.video1_name;
                document.getElementById('video2-name').textContent = info.video2_name;
                if (info.videos && info.videos.length > 2) {
                    document.getElementById('video3-name').textContent = info.videos[2].name;
                }
                if (info.videos && info.videos.length > 3) {
                    document.getElementById('video4-name').textContent = info.videos[3].name;
                }
                
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
                // Fetch frames for all active cameras
                const fetches = [];
                for (let cam = 1; cam <= numCameras; cam++) {
                    fetches.push(fetch(`/api/frame/${cam}/${frameNum}`, { signal: controller.signal }));
                }
                const responses = await Promise.all(fetches);
                const dataArr = await Promise.all(responses.map(r => r.json()));
                
                // Update each camera's display
                for (let cam = 1; cam <= numCameras; cam++) {
                    const data = dataArr[cam - 1];
                    if (data.image) {
                        document.getElementById('frame' + cam).src = 'data:image/jpeg;base64,' + data.image;
                        document.getElementById('loading' + cam).style.display = 'none';
                    }
                    
                    // Update PTS display
                    const ptsEl = document.getElementById('pts' + cam);
                    if (ptsEl) {
                        ptsEl.textContent = data.pts_us ? `PTS: ${(data.pts_us/1000).toFixed(1)}ms` : 'PTS: —';
                        ptsEl.classList.toggle('none', !data.pts_us);
                    }
                    
                    // Update QR display
                    const tsEl = document.getElementById('ts' + cam);
                    if (tsEl) {
                        tsEl.textContent = data.timestamp ? `QR: ${data.timestamp}` : 'QR: —';
                        tsEl.classList.toggle('none', !data.timestamp);
                    }
                }
                
                // Use first two cameras for offset calculations (backward compatible)
                const data1 = dataArr[0];
                const data2 = dataArr[1];
                
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
        
        // Analysis functions
        function updateProgress(status) {
            const statusText = document.getElementById('analysis-status-text');
            const progressBar = document.getElementById('analysis-progress');
            const percentText = document.getElementById('analysis-percent');
            
            statusText.textContent = status.status;
            progressBar.style.width = `${status.percent}%`;
            percentText.textContent = `${status.percent}%`;
        }
        
        async function runAnalysis() {
            const sampleInterval = parseInt(document.getElementById('sample-interval').value) || 30;
            const fastMode = document.getElementById('fast-mode').checked;
            const statusDiv = document.getElementById('analysis-status');
            const progressBar = document.getElementById('analysis-progress');
            const percentText = document.getElementById('analysis-percent');
            const resultsDiv = document.getElementById('analysis-results');
            const btn = document.getElementById('analyze-btn');
            const videoBtn = document.getElementById('video-btn');
            
            btn.disabled = true;
            videoBtn.disabled = true;
            btn.textContent = fastMode ? '⚡ Fast Analyzing...' : '⏳ Analyzing...';
            statusDiv.style.display = 'block';
            resultsDiv.style.display = 'none';
            progressBar.style.width = '0%';
            progressBar.style.background = 'linear-gradient(90deg, var(--accent), #00ff88)';
            percentText.textContent = '0%';
            
            try {
                // Start analysis in background
                const startResp = await fetch(`/api/analyze/start?sample_interval=${sampleInterval}&fast=${fastMode}`, { method: 'POST' });
                if (!startResp.ok) {
                    const err = await startResp.json();
                    throw new Error(err.detail || 'Failed to start analysis');
                }
                
                // Poll for progress until complete
                let complete = false;
                while (!complete) {
                    await new Promise(r => setTimeout(r, 250));
                    
                    const statusResp = await fetch('/api/analyze/status');
                    const status = await statusResp.json();
                    
                    updateProgress(status);
                    
                    if (!status.running) {
                        complete = true;
                        if (status.error) {
                            throw new Error(status.error);
                        }
                    }
                }
                
                // Get results
                const resultsResp = await fetch('/api/analyze/results');
                if (!resultsResp.ok) {
                    throw new Error('Failed to get results');
                }
                const data = await resultsResp.json();
                
                // Update statistics
                if (data.statistics.qr_offsets) {
                    document.getElementById('stat-qr-mean').textContent = data.statistics.qr_offsets.mean.toFixed(2) + ' ms';
                    document.getElementById('stat-qr-std').textContent = data.statistics.qr_offsets.std.toFixed(2) + ' ms';
                    document.getElementById('stat-qr-range').textContent = `${data.statistics.qr_offsets.min.toFixed(1)} / ${data.statistics.qr_offsets.max.toFixed(1)} ms`;
                }
                if (data.statistics.sync_errors) {
                    document.getElementById('stat-sync-mean').textContent = data.statistics.sync_errors.mean.toFixed(2) + ' ms';
                }
                if (data.statistics.qr_detection_rate) {
                    document.getElementById('stat-qr-rate').textContent = data.statistics.qr_detection_rate.both.toFixed(1) + '%';
                }
                document.getElementById('stat-samples').textContent = data.sample_count.toLocaleString();
                
                // Load graph
                document.getElementById('sync-graph').src = '/api/analyze/graph?' + Date.now();
                
                progressBar.style.width = '100%';
                percentText.textContent = '100%';
                document.getElementById('analysis-status-text').textContent = 'Analysis complete!';
                resultsDiv.style.display = 'block';
                
                setTimeout(() => { statusDiv.style.display = 'none'; }, 2000);
                
            } catch (e) {
                document.getElementById('analysis-status-text').textContent = 'Error: ' + e.message;
                progressBar.style.background = '#ff4466';
            } finally {
                btn.disabled = false;
                videoBtn.disabled = false;
                btn.textContent = '📈 Generate Analysis & Graph';
            }
        }
        
        async function generateVideo() {
            const sampleInterval = parseInt(document.getElementById('sample-interval').value) || 1;
            const statusDiv = document.getElementById('analysis-status');
            const progressBar = document.getElementById('analysis-progress');
            const percentText = document.getElementById('analysis-percent');
            const videoResultDiv = document.getElementById('video-result');
            const btn = document.getElementById('video-btn');
            const analyzeBtn = document.getElementById('analyze-btn');
            
            btn.disabled = true;
            analyzeBtn.disabled = true;
            btn.textContent = '⏳ Generating Video...';
            statusDiv.style.display = 'block';
            videoResultDiv.style.display = 'none';
            progressBar.style.width = '0%';
            progressBar.style.background = 'linear-gradient(90deg, #ff6600, #ffaa00)';
            percentText.textContent = '0%';
            
            try {
                // Start video generation in background
                // Always output at 30fps and only include frames with QR data in both cameras
                const startResp = await fetch(`/api/generate-video/start?sample_interval=${sampleInterval}&output_fps=30&only_with_qr=true`, { method: 'POST' });
                if (!startResp.ok) {
                    const err = await startResp.json();
                    throw new Error(err.detail || 'Failed to start video generation');
                }
                
                // Poll for progress until complete
                let complete = false;
                while (!complete) {
                    await new Promise(r => setTimeout(r, 500));
                    
                    const statusResp = await fetch('/api/analyze/status');
                    const status = await statusResp.json();
                    
                    updateProgress(status);
                    
                    if (!status.running) {
                        complete = true;
                        if (status.error) {
                            throw new Error(status.error);
                        }
                    }
                }
                
                progressBar.style.width = '100%';
                percentText.textContent = '100%';
                document.getElementById('analysis-status-text').textContent = 'Video generation complete!';
                
                // Get video path
                const videoInfo = await fetch('/api/generate-video/result').then(r => r.json()).catch(() => null);
                if (videoInfo && videoInfo.path) {
                    document.getElementById('video-path').textContent = videoInfo.path;
                } else {
                    document.getElementById('video-path').textContent = 'Video saved to source directory';
                }
                videoResultDiv.style.display = 'block';
                
                setTimeout(() => { statusDiv.style.display = 'none'; }, 2000);
                
            } catch (e) {
                document.getElementById('analysis-status-text').textContent = 'Error: ' + e.message;
                progressBar.style.background = '#ff4466';
            } finally {
                btn.disabled = false;
                analyzeBtn.disabled = false;
                btn.textContent = '🎬 Generate Analysis Video';
            }
        }
        
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
    if cam < 1 or cam > viewer.num_cameras:
        raise HTTPException(status_code=400, detail=f"cam must be 1-{viewer.num_cameras}")
    
    image, timestamp, pts_us = viewer.get_frame(cam, frame_num)
    return JSONResponse(content={'image': image, 'timestamp': timestamp, 'pts_us': pts_us, 'frame': frame_num})


@app.post("/api/load")
async def load_videos(request: VideoLoadRequest):
    global viewer
    
    if viewer:
        viewer.close()
        viewer = None
    
    try:
        viewer = VideoFrameServer(
            request.video1, 
            request.video2,
            request.video3,
            request.video4
        )
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


# Global state for analysis
analysis_state = {
    'running': False,
    'progress': 0,
    'total': 0,
    'status': '',
    'results': None,
    'video_path': None,
    'error': None,
    'operation': None,  # 'analyze' or 'video'
}


def run_analysis_thread(sample_interval: int, fast_mode: bool = True):
    """Run analysis in background thread."""
    global analysis_state
    
    def progress_cb(current, total):
        analysis_state['progress'] = current
        analysis_state['total'] = total
        pct = int((current / total) * 100) if total > 0 else 0
        analysis_state['status'] = f'Analyzing... {pct}% ({current:,} / {total:,} samples)'
    
    try:
        # Always use sequential analysis - threading causes OpenCV WeChat detector crashes
        # fast_mode now just controls whether to skip fallback QR detectors
        results = viewer.analyze_sync(sample_interval=sample_interval, progress_callback=progress_cb, fast_mode=fast_mode)
        
        analysis_state['results'] = results
        analysis_state['status'] = 'Complete!'
        analysis_state['error'] = None
    except Exception as e:
        import traceback
        analysis_state['status'] = f'Error: {str(e)}'
        analysis_state['error'] = str(e)
        print(f"Analysis error: {traceback.format_exc()}")
    finally:
        analysis_state['running'] = False


def run_video_thread(sample_interval: int, max_frames: int, output_fps: float = 30.0, only_with_qr: bool = True):
    """Run video generation in background thread."""
    global analysis_state
    
    video_dir = Path(viewer.video1_path).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"sync_analysis_{timestamp}.mp4"
    output_path = str(video_dir / output_filename)
    
    def progress_cb(current, total, status):
        analysis_state['progress'] = current
        analysis_state['total'] = total
        pct = int((current / total) * 100) if total > 0 else 0
        analysis_state['status'] = f'Generating video... {pct}% ({status})'
    
    try:
        result_path = viewer.generate_analysis_video(
            output_path=output_path,
            sample_interval=sample_interval,
            progress_callback=progress_cb,
            max_frames=max_frames,
            output_fps=output_fps,
            only_with_qr=only_with_qr
        )
        analysis_state['video_path'] = result_path
        analysis_state['status'] = 'Complete!'
        analysis_state['error'] = None
    except Exception as e:
        analysis_state['status'] = f'Error: {str(e)}'
        analysis_state['error'] = str(e)
    finally:
        analysis_state['running'] = False


@app.post("/api/analyze/start")
async def start_analysis(sample_interval: int = 30, fast: bool = True):
    """Start sync analysis in background.
    
    Args:
        sample_interval: Sample every N frames (default: 30)
        fast: Use fast parallel mode (default: True) - uses multiprocessing + WeChat-only QR
    """
    global analysis_state
    
    if not viewer:
        raise HTTPException(status_code=404, detail="No videos loaded")
    
    if analysis_state['running']:
        raise HTTPException(status_code=409, detail="Operation already in progress")
    
    num_workers = min(mp.cpu_count(), 8) if fast else 1
    
    analysis_state['running'] = True
    analysis_state['progress'] = 0
    analysis_state['total'] = viewer.max_frames // sample_interval
    analysis_state['status'] = f'Starting {"fast " if fast else ""}analysis with {num_workers} workers...'
    analysis_state['operation'] = 'analyze'
    analysis_state['error'] = None
    
    # Start in background thread
    thread = threading.Thread(target=run_analysis_thread, args=(sample_interval, fast))
    thread.daemon = True
    thread.start()
    
    return JSONResponse(content={
        'status': 'started', 
        'total_samples': analysis_state['total'],
        'fast_mode': fast,
        'num_workers': num_workers
    })


@app.get("/api/analyze")
async def run_analysis(sample_interval: int = 30):
    """Run sync analysis on loaded videos (blocking for backwards compatibility)."""
    global analysis_state
    
    if not viewer:
        raise HTTPException(status_code=404, detail="No videos loaded")
    
    if analysis_state['running']:
        raise HTTPException(status_code=409, detail="Analysis already in progress")
    
    analysis_state['running'] = True
    analysis_state['progress'] = 0
    analysis_state['status'] = 'Starting analysis...'
    
    def progress_cb(current, total):
        analysis_state['progress'] = current
        analysis_state['total'] = total
        analysis_state['status'] = f'Analyzing frame {current * sample_interval:,} / {total * sample_interval:,}'
    
    try:
        results = viewer.analyze_sync(sample_interval=sample_interval, progress_callback=progress_cb)
        analysis_state['results'] = results
        analysis_state['status'] = 'Complete!'
        
        # Convert numpy types to Python types for JSON serialization
        def convert_stats(stats):
            if stats is None:
                return None
            return {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                    for k, v in stats.items()}
        
        serializable_stats = {}
        for key, value in results['statistics'].items():
            if isinstance(value, dict):
                serializable_stats[key] = convert_stats(value)
            else:
                serializable_stats[key] = value
        
        return JSONResponse(content={
            'status': 'ok',
            'sample_count': results['sample_count'],
            'statistics': serializable_stats,
            'qr_detections': results['qr_detections'],
        })
    except Exception as e:
        analysis_state['status'] = f'Error: {str(e)}'
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        analysis_state['running'] = False


@app.get("/api/analyze/status")
async def get_analysis_status():
    """Get current analysis status."""
    pct = int((analysis_state['progress'] / analysis_state['total']) * 100) if analysis_state['total'] > 0 else 0
    return JSONResponse(content={
        'running': analysis_state['running'],
        'progress': analysis_state['progress'],
        'total': analysis_state['total'],
        'percent': pct,
        'status': analysis_state['status'],
        'operation': analysis_state.get('operation'),
        'has_results': analysis_state['results'] is not None,
        'has_video': analysis_state['video_path'] is not None,
        'error': analysis_state.get('error'),
    })


@app.get("/api/analyze/results")
async def get_analysis_results():
    """Get analysis results after completion."""
    if not analysis_state['results']:
        raise HTTPException(status_code=404, detail="No analysis results available")
    
    results = analysis_state['results']
    
    # Convert numpy types to Python types for JSON serialization
    def convert_stats(stats):
        if stats is None:
            return None
        return {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                for k, v in stats.items()}
    
    serializable_stats = {}
    for key, value in results['statistics'].items():
        if isinstance(value, dict):
            serializable_stats[key] = convert_stats(value)
        else:
            serializable_stats[key] = value
    
    return JSONResponse(content={
        'status': 'ok',
        'sample_count': results['sample_count'],
        'statistics': serializable_stats,
        'qr_detections': results['qr_detections'],
    })


@app.get("/api/analyze/csv")
async def export_analysis_csv():
    """Export analysis results as CSV."""
    if not analysis_state['results']:
        raise HTTPException(status_code=404, detail="No analysis results available. Run analysis first.")
    
    results = analysis_state['results']
    
    # Build CSV content
    lines = ["frame,time_seconds,time_minutes,qr_offset_ms,pts_offset_ms,sync_error_ms"]
    
    fps = results.get('fps', 30)
    
    for i, frame in enumerate(results['frames']):
        time_sec = frame / fps
        time_min = time_sec / 60
        
        qr_off = results['qr_offsets'][i]
        pts_off = results['pts_offsets'][i]
        sync_err = results['sync_errors'][i]
        
        # Format values, use empty string for None
        qr_str = f"{qr_off:.3f}" if qr_off is not None else ""
        pts_str = f"{pts_off:.3f}" if pts_off is not None else ""
        sync_str = f"{sync_err:.3f}" if sync_err is not None else ""
        
        lines.append(f"{frame},{time_sec:.3f},{time_min:.4f},{qr_str},{pts_str},{sync_str}")
    
    csv_content = "\n".join(lines)
    
    # Generate filename based on video names
    video1_name = Path(results.get('video1_name', 'video1')).stem
    video2_name = Path(results.get('video2_name', 'video2')).stem
    filename = f"sync_analysis_{video1_name}_{video2_name}.csv"
    
    return StreamingResponse(
        io.BytesIO(csv_content.encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/api/analyze/graph")
async def get_analysis_graph():
    """Get the sync analysis graph as PNG."""
    if not viewer:
        raise HTTPException(status_code=404, detail="No videos loaded")
    
    if not analysis_state['results']:
        raise HTTPException(status_code=404, detail="No analysis results. Run /api/analyze first.")
    
    if not MATPLOTLIB_AVAILABLE:
        raise HTTPException(status_code=500, detail="matplotlib not available")
    
    try:
        graph_bytes = viewer.generate_sync_graph(analysis_state['results'])
        return StreamingResponse(io.BytesIO(graph_bytes), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-video/start")
async def start_video_generation(sample_interval: int = 1, max_frames: int = None, 
                                  output_fps: float = 30.0, only_with_qr: bool = True):
    """Start video generation in background.
    
    Args:
        sample_interval: Sample every N frames from source (default: 1)
        max_frames: Maximum source frames to process (default: all)
        output_fps: Output video FPS for smooth playback (default: 30)
        only_with_qr: Only include frames with QR data in both cameras (default: True)
    """
    global analysis_state
    
    if not viewer:
        raise HTTPException(status_code=404, detail="No videos loaded")
    
    if analysis_state['running']:
        raise HTTPException(status_code=409, detail="Operation already in progress")
    
    total = max_frames or viewer.max_frames
    analysis_state['running'] = True
    analysis_state['progress'] = 0
    analysis_state['total'] = total
    analysis_state['status'] = 'Starting video generation...'
    analysis_state['operation'] = 'video'
    analysis_state['error'] = None
    analysis_state['video_path'] = None
    
    # Start in background thread
    thread = threading.Thread(target=run_video_thread, args=(sample_interval, max_frames, output_fps, only_with_qr))
    thread.daemon = True
    thread.start()
    
    return JSONResponse(content={'status': 'started', 'total_frames': total, 'output_fps': output_fps, 'only_with_qr': only_with_qr})


@app.get("/api/generate-video/result")
async def get_video_result():
    """Get the generated video path."""
    if not analysis_state['video_path']:
        raise HTTPException(status_code=404, detail="No video generated yet")
    
    return JSONResponse(content={
        'status': 'ok',
        'path': analysis_state['video_path'],
        'filename': Path(analysis_state['video_path']).name,
    })


@app.get("/api/generate-video")
async def generate_video(sample_interval: int = 1, max_frames: int = None):
    """Generate analysis video (blocking for backwards compatibility)."""
    global analysis_state
    
    if not viewer:
        raise HTTPException(status_code=404, detail="No videos loaded")
    
    if analysis_state['running']:
        raise HTTPException(status_code=409, detail="Operation already in progress")
    
    analysis_state['running'] = True
    analysis_state['progress'] = 0
    analysis_state['status'] = 'Starting video generation...'
    
    # Generate output path in same directory as source videos
    video_dir = Path(viewer.video1_path).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"sync_analysis_{timestamp}.mp4"
    output_path = str(video_dir / output_filename)
    
    def progress_cb(current, total, status):
        analysis_state['progress'] = current
        analysis_state['total'] = total
        analysis_state['status'] = status
    
    try:
        result_path = viewer.generate_analysis_video(
            output_path=output_path,
            sample_interval=sample_interval,
            progress_callback=progress_cb,
            max_frames=max_frames
        )
        analysis_state['video_path'] = result_path
        analysis_state['status'] = 'Video generation complete!'
        
        return JSONResponse(content={
            'status': 'ok',
            'path': result_path,
            'filename': output_filename,
        })
    except Exception as e:
        analysis_state['status'] = f'Error: {str(e)}'
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        analysis_state['running'] = False


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
