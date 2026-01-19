#!/usr/bin/env python3
"""
V4L2 Camera Settings Web Server with Live View

Provides a web interface to adjust camera settings and view live feeds.
Usage: python3 camera_settings_server.py [--port 8080]
"""

from flask import Flask, render_template_string, jsonify, request, Response
import subprocess
import re
import threading
import time

app = Flask(__name__)

# Global camera streams
camera_streams = {}
stream_lock = threading.Lock()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>V4L2 Camera Settings</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 20px;
            color: #e4e4e4;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            color: #00d4ff;
            font-size: 2rem;
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        }
        .camera-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 20px;
        }
        .camera-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }
        .camera-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .camera-icon {
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, #00d4ff, #0099cc);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
        }
        .camera-info h2 {
            font-size: 1.1rem;
            color: #fff;
        }
        .camera-info .device {
            font-size: 0.85rem;
            color: #888;
            font-family: monospace;
        }
        .live-view-container {
            margin-bottom: 20px;
            border-radius: 12px;
            overflow: hidden;
            background: #000;
            position: relative;
        }
        .live-view {
            width: 100%;
            height: auto;
            display: block;
            min-height: 200px;
        }
        .live-view-placeholder {
            width: 100%;
            height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
            font-size: 1.1rem;
        }
        .stream-controls {
            display: flex;
            gap: 10px;
            margin-bottom: 16px;
        }
        .stream-btn {
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .stream-btn.start {
            background: #00c853;
            color: #000;
        }
        .stream-btn.stop {
            background: #ff5252;
            color: #fff;
        }
        .stream-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .controls-section {
            max-height: 400px;
            overflow-y: auto;
            padding-right: 10px;
        }
        .controls-section::-webkit-scrollbar {
            width: 6px;
        }
        .controls-section::-webkit-scrollbar-track {
            background: rgba(255,255,255,0.05);
            border-radius: 3px;
        }
        .controls-section::-webkit-scrollbar-thumb {
            background: rgba(255,255,255,0.2);
            border-radius: 3px;
        }
        .control-group {
            margin-bottom: 16px;
        }
        .control-label {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .control-name {
            font-size: 0.9rem;
            color: #ccc;
        }
        .control-value {
            font-family: monospace;
            background: rgba(0, 212, 255, 0.2);
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.85rem;
            color: #00d4ff;
            min-width: 60px;
            text-align: center;
        }
        .slider-container {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .range-label {
            font-size: 0.75rem;
            color: #666;
            min-width: 45px;
        }
        .range-label.max {
            text-align: right;
        }
        input[type="range"] {
            flex: 1;
            height: 6px;
            -webkit-appearance: none;
            appearance: none;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            outline: none;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 18px;
            height: 18px;
            background: linear-gradient(135deg, #00d4ff, #0099cc);
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0, 212, 255, 0.4);
            transition: transform 0.2s;
        }
        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.2);
        }
        select {
            width: 100%;
            padding: 10px 12px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            color: #fff;
            font-size: 0.9rem;
            cursor: pointer;
            outline: none;
        }
        select:focus {
            border-color: #00d4ff;
        }
        select option {
            background: #1a1a2e;
            color: #fff;
        }
        .btn-group {
            display: flex;
            gap: 10px;
            margin-top: 20px;
            padding-top: 16px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        .btn {
            flex: 1;
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-primary {
            background: linear-gradient(135deg, #00d4ff, #0099cc);
            color: #000;
        }
        .btn-primary:hover {
            box-shadow: 0 4px 20px rgba(0, 212, 255, 0.4);
            transform: translateY(-2px);
        }
        .btn-secondary {
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.15);
        }
        .status {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 0.9rem;
            opacity: 0;
            transform: translateY(20px);
            transition: all 0.3s;
            z-index: 1000;
        }
        .status.show {
            opacity: 1;
            transform: translateY(0);
        }
        .status.success {
            background: rgba(0, 200, 83, 0.9);
            color: #fff;
        }
        .status.error {
            background: rgba(255, 82, 82, 0.9);
            color: #fff;
        }
        .no-cameras {
            text-align: center;
            padding: 60px;
            color: #888;
        }
        .no-cameras h2 {
            margin-bottom: 10px;
            color: #ff5252;
        }
        .refresh-btn {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #fff;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.2s;
        }
        .refresh-btn:hover {
            background: rgba(255, 255, 255, 0.15);
        }
        .section-title {
            font-size: 0.9rem;
            color: #00d4ff;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>V4L2 Camera Settings</h1>
        <button class="refresh-btn" onclick="location.reload()">Refresh</button>
        <div id="cameras" class="camera-grid"></div>
    </div>
    <div id="status" class="status"></div>

    <script>
        let cameras = {};
        let activeStreams = {};

        async function loadCameras() {
            try {
                const response = await fetch('/api/cameras');
                cameras = await response.json();
                renderCameras();
            } catch (error) {
                document.getElementById('cameras').innerHTML = `
                    <div class="no-cameras">
                        <h2>Error Loading Cameras</h2>
                        <p>${error.message}</p>
                    </div>
                `;
            }
        }

        function renderCameras() {
            const container = document.getElementById('cameras');
            
            if (Object.keys(cameras).length === 0) {
                container.innerHTML = `
                    <div class="no-cameras">
                        <h2>No USB 2.0 Cameras Found</h2>
                        <p>Make sure your cameras are connected and try refreshing.</p>
                    </div>
                `;
                return;
            }

            container.innerHTML = Object.entries(cameras).map(([device, cam], index) => `
                <div class="camera-card" data-device="${device}">
                    <div class="camera-header">
                        <div class="camera-icon">📷</div>
                        <div class="camera-info">
                            <h2>${cam.name} (Camera ${index + 1})</h2>
                            <div class="device">${device}</div>
                        </div>
                    </div>
                    
                    <div class="section-title">Live View</div>
                    <div class="live-view-container">
                        <img id="stream-${device.replace(/\\//g, '-')}" 
                             class="live-view" 
                             style="display: none;"
                             alt="Live view">
                        <div id="placeholder-${device.replace(/\\//g, '-')}" class="live-view-placeholder">
                            Click "Start Stream" to view
                        </div>
                    </div>
                    <div class="stream-controls">
                        <button class="stream-btn start" onclick="startStream('${device}')">▶ Start Stream</button>
                        <button class="stream-btn stop" onclick="stopStream('${device}')">⬛ Stop Stream</button>
                    </div>
                    
                    <div class="section-title">Controls</div>
                    <div class="controls-section">
                        ${renderControls(device, cam.controls)}
                    </div>
                    <div class="btn-group">
                        <button class="btn btn-secondary" onclick="resetDefaults('${device}')">Reset Defaults</button>
                        <button class="btn btn-primary" onclick="applySettings('${device}')">Apply</button>
                    </div>
                </div>
            `).join('');
        }

        function renderControls(device, controls) {
            return Object.entries(controls).map(([name, ctrl]) => {
                if (ctrl.type === 'int') {
                    return `
                        <div class="control-group">
                            <div class="control-label">
                                <span class="control-name">${formatName(name)}</span>
                                <span class="control-value" id="${device}-${name}-value">${ctrl.value}</span>
                            </div>
                            <div class="slider-container">
                                <span class="range-label">${ctrl.min}</span>
                                <input type="range" 
                                    id="${device}-${name}" 
                                    min="${ctrl.min}" 
                                    max="${ctrl.max}" 
                                    step="${ctrl.step || 1}"
                                    value="${ctrl.value}"
                                    data-device="${device}"
                                    data-control="${name}"
                                    oninput="updateValue(this)">
                                <span class="range-label max">${ctrl.max}</span>
                            </div>
                        </div>
                    `;
                } else if (ctrl.type === 'menu' || ctrl.type === 'bool') {
                    const options = ctrl.options || {0: 'Off', 1: 'On'};
                    return `
                        <div class="control-group">
                            <div class="control-label">
                                <span class="control-name">${formatName(name)}</span>
                            </div>
                            <select id="${device}-${name}" 
                                data-device="${device}" 
                                data-control="${name}"
                                onchange="updateSelectValue(this)">
                                ${Object.entries(options).map(([val, label]) => 
                                    `<option value="${val}" ${ctrl.value == val ? 'selected' : ''}>${label}</option>`
                                ).join('')}
                            </select>
                        </div>
                    `;
                }
                return '';
            }).join('');
        }

        function formatName(name) {
            return name.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
        }

        function updateValue(input) {
            const valueSpan = document.getElementById(`${input.dataset.device}-${input.dataset.control}-value`);
            if (valueSpan) {
                valueSpan.textContent = input.value;
            }
            // Auto-apply on change
            autoApplySetting(input.dataset.device, input.dataset.control, input.value);
        }

        function updateSelectValue(select) {
            // Auto-apply on change
            autoApplySetting(select.dataset.device, select.dataset.control, select.value);
        }

        // Debounce to avoid too many requests when dragging sliders
        let applyTimers = {};
        async function autoApplySetting(device, control, value) {
            const key = `${device}-${control}`;
            if (applyTimers[key]) {
                clearTimeout(applyTimers[key]);
            }
            applyTimers[key] = setTimeout(async () => {
                try {
                    const response = await fetch('/api/set', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({device, settings: {[control]: parseInt(value)}})
                    });
                    const result = await response.json();
                    if (!result.success) {
                        showStatus('Error: ' + result.error, 'error');
                    }
                } catch (error) {
                    showStatus('Failed to apply', 'error');
                }
            }, 50);  // 50ms debounce
        }

        function startStream(device) {
            const safeId = device.replace(/\\//g, '-');
            const img = document.getElementById(`stream-${safeId}`);
            const placeholder = document.getElementById(`placeholder-${safeId}`);
            
            // Add timestamp to prevent caching
            img.src = `/api/stream?device=${encodeURIComponent(device)}&t=${Date.now()}`;
            img.style.display = 'block';
            placeholder.style.display = 'none';
            activeStreams[device] = true;
            
            img.onerror = function() {
                showStatus('Failed to start stream', 'error');
                stopStream(device);
            };
        }

        function stopStream(device) {
            const safeId = device.replace(/\\//g, '-');
            const img = document.getElementById(`stream-${safeId}`);
            const placeholder = document.getElementById(`placeholder-${safeId}`);
            
            img.src = '';
            img.style.display = 'none';
            placeholder.style.display = 'flex';
            delete activeStreams[device];
        }

        async function applySettings(device) {
            const card = document.querySelector(`[data-device="${device}"]`);
            const inputs = card.querySelectorAll('input[type="range"], select');
            const settings = {};
            
            inputs.forEach(input => {
                settings[input.dataset.control] = parseInt(input.value);
            });

            try {
                const response = await fetch('/api/set', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({device, settings})
                });
                const result = await response.json();
                
                if (result.success) {
                    showStatus('Settings applied!', 'success');
                } else {
                    showStatus('Error: ' + result.error, 'error');
                }
            } catch (error) {
                showStatus('Failed to apply settings', 'error');
            }
        }

        async function resetDefaults(device) {
            try {
                const response = await fetch('/api/reset', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({device})
                });
                const result = await response.json();
                
                if (result.success) {
                    showStatus('Reset to defaults!', 'success');
                    loadCameras();
                } else {
                    showStatus('Error: ' + result.error, 'error');
                }
            } catch (error) {
                showStatus('Failed to reset', 'error');
            }
        }

        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = `status ${type} show`;
            setTimeout(() => {
                status.classList.remove('show');
            }, 3000);
        }

        loadCameras();
    </script>
</body>
</html>
"""


def detect_usb20_cameras():
    """Detect USB 2.0 cameras, returns dict of device -> name"""
    try:
        result = subprocess.run(
            ['v4l2-ctl', '--list-devices'],
            capture_output=True, text=True
        )
        output = result.stdout
    except FileNotFoundError:
        return {}
    
    cameras = {}
    lines = output.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("USB 2.0 Camera"):
            name = line.strip().rstrip(':')
            i += 1
            while i < len(lines) and lines[i].startswith('\t'):
                match = re.search(r'/dev/video\d+', lines[i])
                if match:
                    cameras[match.group()] = name
                    break
                i += 1
        else:
            i += 1
    
    return cameras


def get_camera_controls(device):
    """Get all controls for a camera device"""
    result = subprocess.run(
        ['v4l2-ctl', '-d', device, '--list-ctrls-menus'],
        capture_output=True, text=True
    )
    
    controls = {}
    lines = result.stdout.split('\n')
    current_control = None
    
    for line in lines:
        ctrl_match = re.match(
            r'\s*(\w+)\s+0x[0-9a-f]+\s+\((\w+)\)\s*:\s*(.*)',
            line
        )
        
        if ctrl_match:
            name = ctrl_match.group(1)
            ctrl_type = ctrl_match.group(2)
            params_str = ctrl_match.group(3)
            
            control = {'type': ctrl_type, 'name': name}
            
            if ctrl_type == 'int':
                for param in ['min', 'max', 'step', 'default', 'value']:
                    match = re.search(rf'{param}=(-?\d+)', params_str)
                    if match:
                        control[param] = int(match.group(1))
            elif ctrl_type == 'bool':
                match = re.search(r'default=(\d+)', params_str)
                if match:
                    control['default'] = int(match.group(1))
                match = re.search(r'value=(\d+)', params_str)
                if match:
                    control['value'] = int(match.group(1))
                control['options'] = {0: 'Off', 1: 'On'}
            elif ctrl_type == 'menu':
                match = re.search(r'default=(\d+)', params_str)
                if match:
                    control['default'] = int(match.group(1))
                match = re.search(r'value=(\d+)', params_str)
                if match:
                    control['value'] = int(match.group(1))
                control['options'] = {}
                current_control = name
                controls[name] = control
                continue
            
            controls[name] = control
            current_control = name if ctrl_type == 'menu' else None
        
        elif current_control and line.strip():
            menu_match = re.match(r'\s+(\d+):\s+(.+)', line)
            if menu_match:
                idx = int(menu_match.group(1))
                label = menu_match.group(2).strip()
                controls[current_control]['options'][idx] = label
    
    return controls


def generate_mjpeg_stream(device):
    """Generate MJPEG stream from camera using GStreamer"""
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    
    Gst.init(None)
    
    # GStreamer pipeline: capture MJPEG, output to appsink
    pipeline_str = f"""
        v4l2src device={device} ! 
        image/jpeg,width=640,height=480,framerate=30/1 ! 
        appsink name=sink emit-signals=false max-buffers=1 drop=true
    """
    
    pipeline = Gst.parse_launch(pipeline_str)
    sink = pipeline.get_by_name('sink')
    
    pipeline.set_state(Gst.State.PLAYING)
    
    try:
        while True:
            sample = sink.emit('pull-sample')
            if sample:
                buffer = sample.get_buffer()
                success, map_info = buffer.map(Gst.MapFlags.READ)
                if success:
                    frame_data = bytes(map_info.data)
                    buffer.unmap(map_info)
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            else:
                time.sleep(0.01)
    except GeneratorExit:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/cameras')
def get_cameras():
    cameras = detect_usb20_cameras()
    result = {}
    
    for device, name in cameras.items():
        controls = get_camera_controls(device)
        result[device] = {
            'name': name,
            'controls': controls
        }
    
    return jsonify(result)


@app.route('/api/stream')
def video_stream():
    device = request.args.get('device', '/dev/video0')
    return Response(
        generate_mjpeg_stream(device),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/set', methods=['POST'])
def set_controls():
    data = request.json
    device = data.get('device')
    settings = data.get('settings', {})
    
    if not device:
        return jsonify({'success': False, 'error': 'No device specified'})
    
    try:
        cmd = ['v4l2-ctl', '-d', device]
        for control, value in settings.items():
            cmd.extend(['-c', f'{control}={value}'])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return jsonify({'success': False, 'error': result.stderr})
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/reset', methods=['POST'])
def reset_controls():
    data = request.json
    device = data.get('device')
    
    if not device:
        return jsonify({'success': False, 'error': 'No device specified'})
    
    try:
        controls = get_camera_controls(device)
        cmd = ['v4l2-ctl', '-d', device]
        
        for name, ctrl in controls.items():
            if 'default' in ctrl:
                cmd.extend(['-c', f'{name}={ctrl["default"]}'])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return jsonify({'success': False, 'error': result.stderr})
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='V4L2 Camera Settings Web Server')
    parser.add_argument('--port', type=int, default=8080, help='Port to run server on')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    args = parser.parse_args()
    
    print(f"\n{'='*50}")
    print("V4L2 Camera Settings Server")
    print(f"{'='*50}")
    print(f"\nStarting server at http://{args.host}:{args.port}")
    print(f"Open http://localhost:{args.port} in your browser\n")
    
    # Use threaded mode for streaming
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
