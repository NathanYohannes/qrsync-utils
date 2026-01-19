#!/usr/bin/env python3
"""
Dual USB 2.0 Camera Recording with Synchronized Capture

Records from two USB 2.0 cameras simultaneously. Atlas cameras are ignored.
Automatically configures camera settings before recording.

Usage: python3 dual_camera_record.py [output_dir] [--fps 30|60|120] [-v]
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import subprocess
import re
import sys
import os
import signal
import json
import urllib.request
import urllib.error
from datetime import datetime

Gst.init(None)


def detect_usb20_cameras():
    """
    Detect USB 2.0 cameras using v4l2-ctl, ignoring Atlas cameras.
    Returns a list of /dev/video* device paths.
    """
    try:
        result = subprocess.run(
            ['v4l2-ctl', '--list-devices'],
            capture_output=True, text=True
        )
        output = result.stdout
    except FileNotFoundError:
        print("Error: v4l2-ctl not found. Install v4l-utils package.")
        sys.exit(1)
    
    usb_cameras = []
    lines = output.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        # Look for "USB 2.0 Camera" entries (not Atlas)
        if line.startswith("USB 2.0 Camera"):
            # Next line(s) contain the device paths
            i += 1
            while i < len(lines) and lines[i].startswith('\t'):
                match = re.search(r'/dev/video\d+', lines[i])
                if match:
                    usb_cameras.append(match.group())
                    break  # Take only first device per camera
                i += 1
        else:
            i += 1
    
    return usb_cameras


def get_camera_settings(device):
    """
    Read current camera settings using v4l2-ctl.
    Returns a dict of control_name -> value.
    """
    result = subprocess.run(
        ['v4l2-ctl', '-d', device, '--list-ctrls'],
        capture_output=True, text=True
    )
    
    settings = {}
    for line in result.stdout.split('\n'):
        # Match: control_name 0x00xxxxxx (type) : ... value=N
        match = re.match(r'\s*(\w+)\s+0x[0-9a-f]+\s+\(\w+\)\s*:.*value=(-?\d+)', line)
        if match:
            settings[match.group(1)] = int(match.group(2))
    
    return settings


def apply_camera_settings(device, settings):
    """
    Apply camera settings using v4l2-ctl.
    """
    cmd = ['v4l2-ctl', '-d', device]
    for control, value in settings.items():
        cmd.extend(['-c', f'{control}={value}'])
    
    subprocess.run(cmd, capture_output=True)


# Static settings derived from web viewer tuning
STATIC_CAMERA_SETTINGS = {
    'brightness': -15,              # Reduced brightness for QR readability
    'contrast': 32,                 # Default contrast
    'saturation': 64,               # Default saturation
    'hue': 0,                       # Default hue
    'gamma': 72,                    # Default gamma
    'gain': 0,                      # No gain - reduces noise
    'sharpness': 5,                 # Default sharpness
    'backlight_compensation': 0,    # Disabled for consistent exposure
    'auto_exposure': 3,             # Aperture Priority Mode
    'exposure_time_absolute': 32,   # Fast exposure for motion/QR codes
    'exposure_dynamic_framerate': 0,# Dynamic framerate OFF for consistent timing
    'white_balance_automatic': 1,   # Auto white balance
    'power_line_frequency': 1,      # 50 Hz
}


def apply_static_settings(cameras):
    """
    Apply statically defined camera settings derived from web viewer tuning.
    """
    print("Applying static camera settings...")
    
    for cam_device in cameras:
        apply_camera_settings(cam_device, STATIC_CAMERA_SETTINGS)
        
        # Print key settings
        key_settings = ['brightness', 'gain', 'exposure_time_absolute', 'auto_exposure']
        summary = ', '.join(f"{k}={STATIC_CAMERA_SETTINGS.get(k)}" for k in key_settings)
        print(f"  {cam_device}: {summary}")
    
    print("✓ Static camera settings applied")


def fetch_settings_from_webapp(server_url="http://localhost:8080"):
    """
    Fetch current camera settings from the web app API.
    Returns dict of device -> {control: value, ...}
    """
    try:
        url = f"{server_url}/api/cameras"
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
            
            # Extract just the control values for each device
            result = {}
            for device, cam_data in data.items():
                controls = cam_data.get('controls', {})
                result[device] = {
                    name: ctrl.get('value') 
                    for name, ctrl in controls.items() 
                    if 'value' in ctrl
                }
            return result
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
        print(f"  Warning: Could not fetch from web app ({e})")
        return None


def read_and_apply_current_settings(cameras, server_url="http://localhost:8080"):
    """
    Fetch settings from the web app and apply them to cameras.
    Falls back to reading directly from cameras if web app is unavailable.
    """
    print("Fetching camera settings from web app...")
    
    # Try to get settings from the web app first
    webapp_settings = fetch_settings_from_webapp(server_url)
    
    if webapp_settings:
        print("  ✓ Connected to camera settings server")
        for cam_device in cameras:
            if cam_device in webapp_settings:
                settings = webapp_settings[cam_device]
                apply_camera_settings(cam_device, settings)
                
                # Print key settings
                key_settings = ['brightness', 'contrast', 'gain', 'exposure_time_absolute', 'auto_exposure']
                summary = ', '.join(f"{k}={settings.get(k, '?')}" for k in key_settings if k in settings)
                print(f"  {cam_device}: {summary}")
            else:
                print(f"  {cam_device}: Not found in web app, reading directly...")
                settings = get_camera_settings(cam_device)
                apply_camera_settings(cam_device, settings)
    else:
        print("  Web app not available, reading settings directly from cameras...")
        for cam_device in cameras:
            settings = get_camera_settings(cam_device)
            apply_camera_settings(cam_device, settings)
            
            key_settings = ['brightness', 'contrast', 'gain', 'exposure_time_absolute', 'auto_exposure']
            summary = ', '.join(f"{k}={settings.get(k, '?')}" for k in key_settings if k in settings)
            print(f"  {cam_device}: {summary}")
    
    print("✓ Camera settings applied")


def get_next_rec_number(base_dir):
    """Find the next available rec# folder number."""
    rec_num = 1
    while os.path.exists(os.path.join(base_dir, f"rec{rec_num}")):
        rec_num += 1
    return rec_num


class DualCameraRecorder:
    def __init__(self, output_dir=".", fps=30, verbose=False, server_url="http://localhost:8080"):
        self.base_output_dir = output_dir
        self.fps = fps
        self.verbose = verbose
        self.server_url = server_url
        
        # Find next rec# and create folder
        self.rec_num = get_next_rec_number(self.base_output_dir)
        self.output_dir = os.path.join(self.base_output_dir, f"rec{self.rec_num}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Camera settings
        self.cam_width = 640
        self.cam_height = 480
        
        self.pipeline = None
        self.loop = None
        
        # Frame counters for verbose mode
        self.cam1_frame_count = 0
        self.cam2_frame_count = 0
        
        # Detect cameras
        self.cameras = detect_usb20_cameras()
    
    def on_new_sample_cam1(self, appsink):
        """Callback for camera 1 - prints timestamps in verbose mode"""
        sample = appsink.emit("pull-sample")
        if sample:
            buffer = sample.get_buffer()
            pts = buffer.pts
            dts = buffer.dts
            duration = buffer.duration
            
            self.cam1_frame_count += 1
            
            pts_sec = pts / 1e9 if pts != Gst.CLOCK_TIME_NONE else -1
            dts_sec = dts / 1e9 if dts != Gst.CLOCK_TIME_NONE else -1
            dur_ms = duration / 1e6 if duration != Gst.CLOCK_TIME_NONE else -1
            
            print(f"CAM1 [{self.cam1_frame_count:6d}] PTS: {pts_sec:10.4f}s  DTS: {dts_sec:10.4f}s  Duration: {dur_ms:6.2f}ms")
        
        return Gst.FlowReturn.OK
    
    def on_new_sample_cam2(self, appsink):
        """Callback for camera 2 - prints timestamps in verbose mode"""
        sample = appsink.emit("pull-sample")
        if sample:
            buffer = sample.get_buffer()
            pts = buffer.pts
            dts = buffer.dts
            duration = buffer.duration
            
            self.cam2_frame_count += 1
            
            pts_sec = pts / 1e9 if pts != Gst.CLOCK_TIME_NONE else -1
            dts_sec = dts / 1e9 if dts != Gst.CLOCK_TIME_NONE else -1
            dur_ms = duration / 1e6 if duration != Gst.CLOCK_TIME_NONE else -1
            
            print(f"CAM2 [{self.cam2_frame_count:6d}] PTS: {pts_sec:10.4f}s  DTS: {dts_sec:10.4f}s  Duration: {dur_ms:6.2f}ms")
        
        return Gst.FlowReturn.OK
        
    def build_pipeline(self):
        """Build the GStreamer pipeline for dual camera recording"""
        
        if len(self.cameras) < 2:
            print(f"Error: Found only {len(self.cameras)} USB 2.0 camera(s). Need at least 2.")
            print(f"Detected: {self.cameras}")
            subprocess.run(['v4l2-ctl', '--list-devices'])
            sys.exit(1)
        
        cam1 = self.cameras[0]
        cam2 = self.cameras[1]
        
        # Apply static camera settings derived from web viewer tuning
        apply_static_settings(self.cameras)
        
        video1_path = os.path.join(self.output_dir, f"cam1_rec{self.rec_num}.mkv")
        video2_path = os.path.join(self.output_dir, f"cam2_rec{self.rec_num}.mkv")
        
        # MJPEG PASSTHROUGH: No inter-frame compression, no temporal blur
        # Each frame is independent - perfect for QR code analysis
        
        # do-timestamp=true: Use pipeline clock for timestamps instead of camera timestamps
        # This ensures both cameras are timestamped against the same clock reference
        
        if self.verbose:
            # Save raw MJPEG + print timestamps via appsink
            pipeline_str = f"""
                v4l2src device={cam1} do-timestamp=true ! 
                    image/jpeg,width={self.cam_width},height={self.cam_height},framerate={self.fps}/1 ! 
                    tee name=t1 !
                        queue ! matroskamux ! filesink location={video1_path}
                    t1. ! queue ! 
                        appsink name=sink1 emit-signals=true sync=false
                
                v4l2src device={cam2} do-timestamp=true ! 
                    image/jpeg,width={self.cam_width},height={self.cam_height},framerate={self.fps}/1 ! 
                    tee name=t2 !
                        queue ! matroskamux ! filesink location={video2_path}
                    t2. ! queue ! 
                        appsink name=sink2 emit-signals=true sync=false
            """
        else:
            # Direct MJPEG passthrough - maximum quality, no re-encoding
            pipeline_str = f"""
                v4l2src device={cam1} do-timestamp=true ! 
                    image/jpeg,width={self.cam_width},height={self.cam_height},framerate={self.fps}/1 ! 
                    queue !
                    matroskamux ! 
                    filesink location={video1_path}
                
                v4l2src device={cam2} do-timestamp=true ! 
                    image/jpeg,width={self.cam_width},height={self.cam_height},framerate={self.fps}/1 ! 
                    queue !
                    matroskamux ! 
                    filesink location={video2_path}
            """
        
        print(f"Recording to: {self.output_dir}/")
        print(f"  Camera 1 ({cam1}) -> cam1_rec{self.rec_num}.mkv")
        print(f"  Camera 2 ({cam2}) -> cam2_rec{self.rec_num}.mkv")
        print(f"Resolution: {self.cam_width}x{self.cam_height} @ {self.fps}fps")
        print(f"Codec: MJPEG passthrough (no inter-frame blur)")
        if self.verbose:
            print("Verbose mode: ON (printing frame timestamps)")
        
        self.pipeline = Gst.parse_launch(pipeline_str)
        
        # Connect appsink callbacks for verbose mode
        if self.verbose:
            sink1 = self.pipeline.get_by_name("sink1")
            sink1.connect("new-sample", self.on_new_sample_cam1)
            
            sink2 = self.pipeline.get_by_name("sink2")
            sink2.connect("new-sample", self.on_new_sample_cam2)
        
        self.video1_path = video1_path
        self.video2_path = video2_path
        
    def on_message(self, bus, message):
        """Handle pipeline messages"""
        t = message.type
        if t == Gst.MessageType.EOS:
            print("\nEnd of stream")
            self.loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}")
            if debug:
                print(f"Debug: {debug}")
            self.loop.quit()
        elif t == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            print(f"Warning: {warn}")
    
    def stop(self):
        """Stop recording gracefully"""
        if self.pipeline:
            print("\nStopping recording...")
            self.pipeline.send_event(Gst.Event.new_eos())
    
    def run(self):
        """Start recording"""
        self.build_pipeline()
        
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message)
        
        self.loop = GLib.MainLoop()
        
        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        print("\n" + "="*50)
        print("Recording started. Press Ctrl+C to stop.")
        print("="*50)
        print("\nSynchronizing cameras...")
        
        # SYNCHRONIZATION STRATEGY:
        # Both cameras in the same pipeline automatically share the same clock.
        # Transition through PAUSED ensures both are ready before starting.
        
        self.pipeline.set_state(Gst.State.PAUSED)
        
        # Wait for both cameras to reach PAUSED state (ready to capture)
        ret = self.pipeline.get_state(timeout=5 * Gst.SECOND)
        if ret[0] == Gst.StateChangeReturn.FAILURE:
            print("Failed to pause pipeline - check camera connections")
            sys.exit(1)
        
        print("Cameras synchronized. Starting recording...\n")
        
        # Transition to PLAYING - both cameras start together
        self.pipeline.set_state(Gst.State.PLAYING)
        
        try:
            self.loop.run()
        except Exception as e:
            print(f"Error: {e}")
        
        import time
        time.sleep(0.5)  # Allow EOS to propagate
        
        self.pipeline.set_state(Gst.State.NULL)
        
        print("\nRecording complete!")
        if self.verbose:
            print(f"Total frames - CAM1: {self.cam1_frame_count}, CAM2: {self.cam2_frame_count}")
        print(f"Files saved to: {self.output_dir}/")
        print(f"  - cam1_rec{self.rec_num}.mkv")
        print(f"  - cam2_rec{self.rec_num}.mkv")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Record from two USB 2.0 cameras simultaneously')
    parser.add_argument('output_dir', nargs='?', default='.', help='Output directory (default: current)')
    parser.add_argument('--fps', type=int, default=30, choices=[30, 60, 120, 180],
                        help='Framerate (default: 30)')
    parser.add_argument('--width', type=int, default=640, help='Frame width (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Frame height (default: 480)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print frame timestamps (PTS/DTS) to console')
    parser.add_argument('--server', default='http://localhost:8080',
                        help='Camera settings server URL (default: http://localhost:8080)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*50)
    print("Dual USB 2.0 Camera Recording")
    print("="*50)
    print("Searching for USB 2.0 cameras (ignoring Atlas)...")
    
    recorder = DualCameraRecorder(args.output_dir, args.fps, args.verbose, args.server)
    recorder.cam_width = args.width
    recorder.cam_height = args.height
    
    print(f"Found {len(recorder.cameras)} USB 2.0 cameras: {recorder.cameras}")
    
    recorder.run()


if __name__ == "__main__":
    main()
