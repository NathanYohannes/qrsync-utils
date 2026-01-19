#!/usr/bin/env python3
"""
Camera Sync Analyzer - Analyzes frame synchronization and generates timeline visualizations.
Combines sync analysis and visualization into a single tool.
"""

import argparse
import sys
import csv
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt

# QR Detection
_qr_detector = cv2.QRCodeDetector()
_wechat_detector = None


def _get_wechat_detector():
    global _wechat_detector
    if _wechat_detector is None:
        try:
            _wechat_detector = cv2.wechat_qrcode_WeChatQRCode()
        except:
            _wechat_detector = False
    return _wechat_detector if _wechat_detector else None


def decode_qr(frame: np.ndarray) -> Optional[str]:
    if frame is None:
        return None
    
    wechat = _get_wechat_detector()
    if wechat:
        try:
            results, _ = wechat.detectAndDecode(frame)
            if results and results[0]:
                return results[0]
        except:
            pass
    
    try:
        data, _, _ = _qr_detector.detectAndDecode(frame)
        if data:
            return data
    except:
        pass
    
    return None


def analyze_videos(cam1_path: Path, cam2_path: Path, num_samples: int = 500) -> List[dict]:
    """Analyze frame sync between two videos using sampling."""
    cap1 = cv2.VideoCapture(str(cam1_path))
    cap2 = cv2.VideoCapture(str(cam2_path))
    
    if not cap1.isOpened() or not cap2.isOpened():
        print(f"Error: Could not open video files")
        sys.exit(1)
    
    # Get frame counts
    count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if count1 <= 0 or count1 > 10000000:
        count1 = 216000
    if count2 <= 0 or count2 > 10000000:
        count2 = 216000
    
    max_frame = min(count1, count2) - 1
    sample_frames = np.linspace(0, max_frame, num_samples, dtype=int).tolist()
    
    print(f"Analyzing {cam1_path.name} vs {cam2_path.name}")
    print(f"Sampling {num_samples} frames from 0 to {max_frame:,}")
    
    matches = []
    
    for i, frame_num in enumerate(sample_frames):
        # Read cam1
        cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret1, frame1 = cap1.read()
        ts1 = decode_qr(frame1) if ret1 else None
        
        # Read cam2
        cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret2, frame2 = cap2.read()
        ts2 = decode_qr(frame2) if ret2 else None
        
        if ts1 and ts2:
            try:
                drift_us = int(ts2) - int(ts1)
                matches.append({
                    'frame': frame_num,
                    'cam1_timestamp_us': int(ts1),
                    'cam2_timestamp_us': int(ts2),
                    'drift_us': drift_us,
                    'drift_ms': drift_us / 1000.0
                })
            except ValueError:
                pass
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{num_samples} samples, {len(matches)} matches")
    
    cap1.release()
    cap2.release()
    
    print(f"Complete: {len(matches)} matching frames with QR codes")
    return matches


def save_csv(matches: List[dict], output_path: Path):
    """Save analysis results to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'cam1_timestamp_us', 'cam2_timestamp_us', 'drift_us', 'drift_ms'])
        writer.writeheader()
        writer.writerows(matches)
    print(f"Saved CSV: {output_path}")


def generate_visualizations(matches: List[dict], output_base: Path):
    """Generate timeline visualization images."""
    if not matches:
        print("No data to visualize")
        return
    
    frames = np.array([m['frame'] for m in matches])
    cam1_ts = np.array([m['cam1_timestamp_us'] for m in matches])
    cam2_ts = np.array([m['cam2_timestamp_us'] for m in matches])
    drifts = np.array([m['drift_ms'] for m in matches])
    
    # Convert to relative time
    start_time = min(cam1_ts[0], cam2_ts[0])
    cam1_time = (cam1_ts - start_time) / 1_000_000
    cam2_time = (cam2_ts - start_time) / 1_000_000
    
    # Calculate stats
    duration = cam1_time[-1] - cam1_time[0]
    initial_drift = drifts[0]
    final_drift = drifts[-1]
    drift_change = final_drift - initial_drift
    
    # --- Timeline Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [2, 1.5]})
    fig.patch.set_facecolor('#0a0a0f')
    
    for ax in axes:
        ax.set_facecolor('#12121a')
        ax.tick_params(colors='#8888a0')
        ax.spines['bottom'].set_color('#2a2a3a')
        ax.spines['left'].set_color('#2a2a3a')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Sample points for display
    step = max(1, len(frames) // 30)
    sample_idx = list(range(0, len(frames), step))
    
    # Camera timelines
    ax1 = axes[0]
    ax1.axhline(y=1, color='#00aaff', linewidth=3, alpha=0.7, label='Camera 1')
    ax1.axhline(y=0, color='#ff6688', linewidth=3, alpha=0.7, label='Camera 2')
    
    for idx in sample_idx:
        # Cam1 points
        ax1.scatter(cam1_time[idx], 1, s=80, c='#00aaff', zorder=3)
        ax1.annotate(f'F{frames[idx]:,}\n{int(cam1_ts[idx] % 1000000)}μs', 
                    xy=(cam1_time[idx], 1), xytext=(0, 15), textcoords='offset points',
                    fontsize=7, ha='center', color='#00aaff',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a25', edgecolor='#00aaff', alpha=0.8))
        
        # Cam2 points
        ax2_drift = drifts[idx]
        drift_color = '#00ff88' if abs(ax2_drift) < 20 else '#ffaa00' if abs(ax2_drift) < 50 else '#ff4466'
        ax1.scatter(cam2_time[idx], 0, s=80, c='#ff6688', zorder=3)
        ax1.annotate(f'F{frames[idx]:,}\n{ax2_drift:+.0f}ms', 
                    xy=(cam2_time[idx], 0), xytext=(0, -25), textcoords='offset points',
                    fontsize=7, ha='center', color=drift_color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a25', edgecolor='#ff6688', alpha=0.8))
    
    ax1.set_ylim(-0.8, 1.8)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Camera 2', 'Camera 1'], fontsize=11, fontweight='bold')
    ax1.set_xlabel('Real-World Time (seconds)', fontsize=11, color='#e8e8ed')
    ax1.legend(loc='upper right', facecolor='#1a1a25', edgecolor='#2a2a3a', labelcolor='#e8e8ed')
    ax1.set_title('Camera Timestamp Timelines (Sample Points)', fontsize=14, fontweight='bold', color='#00d4ff', pad=15)
    
    # Stats box
    stats_text = f"Recording Statistics:\n• Duration: {duration:.1f}s (~{duration/60:.1f} min)\n• Frame range: {frames[0]:,} to {frames[-1]:,}\n• Initial drift: {initial_drift:.1f}ms\n• Final drift: {final_drift:.1f}ms\n• Total drift: {drift_change:.1f}ms\n• Samples shown: {len(sample_idx)}/{len(frames)}"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9, verticalalignment='top',
            fontfamily='monospace', color='#e8e8ed',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffcc', edgecolor='#cccc00', alpha=0.9))
    
    # Drift plot
    ax2 = axes[1]
    ax2.plot(cam1_time, drifts, color='#9b59b6', linewidth=2, alpha=0.8)
    ax2.fill_between(cam1_time, 0, drifts, alpha=0.3, color='#9b59b6')
    ax2.axhline(y=0, color='#4a4a5a', linestyle='--', linewidth=1)
    ax2.set_xlabel('Real-World Time (seconds)', fontsize=11, color='#e8e8ed')
    ax2.set_ylabel('Drift (ms)', fontsize=11, color='#e8e8ed')
    ax2.set_title('Timestamp Drift Over Time (Cam2 - Cam1)', fontsize=12, fontweight='bold', color='#9b59b6')
    ax2.grid(True, alpha=0.2, color='#4a4a5a')
    
    plt.tight_layout()
    timeline_path = output_base.parent / (output_base.stem + '_timeline.png')
    plt.savefig(timeline_path, dpi=150, facecolor='#0a0a0f', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Saved: {timeline_path}")
    
    # --- Overlay Plot ---
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#12121a')
    ax.tick_params(colors='#8888a0')
    ax.spines['bottom'].set_color('#2a2a3a')
    ax.spines['left'].set_color('#2a2a3a')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.axhline(y=1, color='#00aaff', linewidth=4, alpha=0.6, label='Camera 1')
    ax.axhline(y=0, color='#ff6688', linewidth=4, alpha=0.6, label='Camera 2')
    
    for idx in sample_idx:
        ax.scatter(cam1_time[idx], 1, s=100, c='#00aaff', zorder=3, edgecolors='white', linewidths=0.5)
        ax.scatter(cam2_time[idx], 0, s=100, c='#ff6688', zorder=3, edgecolors='white', linewidths=0.5)
    
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Camera 2', 'Camera 1'], fontsize=12, fontweight='bold')
    ax.set_xlabel('Real-World Time (seconds)', fontsize=12, color='#e8e8ed')
    ax.set_title('Overlaid Timeline: Both Cameras at Same Frame Numbers', fontsize=14, fontweight='bold', color='#00d4ff')
    ax.legend(loc='upper left', facecolor='#1a1a25', edgecolor='#2a2a3a', labelcolor='#e8e8ed')
    
    # Drift stats box
    drift_stats = f"Drift Analysis:\n• Initial: {initial_drift:.1f}ms @ frame {frames[0]:,}\n• Final: {final_drift:.1f}ms @ frame {frames[-1]:,}\n• Change: {drift_change:.1f}ms over {frames[-1]-frames[0]:,} frames\n• Rate: {drift_change*1000/(frames[-1]-frames[0]):.3f} μs/frame"
    ax.text(0.98, 0.98, drift_stats, transform=ax.transAxes, fontsize=9, verticalalignment='top', ha='right',
            fontfamily='monospace', color='#1a1a25',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffff00', edgecolor='#cccc00', alpha=0.95))
    
    plt.tight_layout()
    overlay_path = output_base.parent / (output_base.stem + '_overlay.png')
    plt.savefig(overlay_path, dpi=150, facecolor='#0a0a0f', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Saved: {overlay_path}")


def print_summary(matches: List[dict]):
    """Print analysis summary."""
    if not matches:
        print("No matches found")
        return
    
    drifts = np.array([m['drift_ms'] for m in matches])
    frames = np.array([m['frame'] for m in matches])
    
    print("\n" + "=" * 60)
    print("SYNC ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Matched frames:    {len(matches)}")
    print(f"Frame range:       {frames[0]:,} → {frames[-1]:,}")
    print(f"Initial drift:     {drifts[0]:+.1f} ms")
    print(f"Final drift:       {drifts[-1]:+.1f} ms")
    print(f"Drift change:      {drifts[-1] - drifts[0]:+.1f} ms")
    print(f"Min/Max drift:     {drifts.min():.1f} / {drifts.max():.1f} ms")
    print(f"Drift range:       {drifts.max() - drifts.min():.1f} ms")
    
    if drifts.max() - drifts.min() > 1000:
        print("\n⚠️  Large drift range detected - possible frame drops!")
    elif abs(drifts[-1] - drifts[0]) < 50:
        print("\n✓  Good sync - minimal drift accumulation")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze camera sync and generate timeline visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_sync.py cam1.mkv cam2.mkv
  python analyze_sync.py cam1.mkv cam2.mkv -o recordings/rec10/rec10
  python analyze_sync.py cam1.mkv cam2.mkv -n 1000  # More samples
        """
    )
    parser.add_argument('cam1', help='Camera 1 video file')
    parser.add_argument('cam2', help='Camera 2 video file')
    parser.add_argument('-o', '--output', help='Output base path (without extension)')
    parser.add_argument('-n', '--samples', type=int, default=500, help='Number of frames to sample (default: 500)')
    
    args = parser.parse_args()
    
    cam1_path = Path(args.cam1).expanduser()
    cam2_path = Path(args.cam2).expanduser()
    
    if not cam1_path.exists():
        print(f"Error: {cam1_path} not found")
        sys.exit(1)
    if not cam2_path.exists():
        print(f"Error: {cam2_path} not found")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_base = Path(args.output)
    else:
        output_base = Path.cwd() / f"sync_{cam1_path.stem}_{cam2_path.stem}"
    
    output_base.parent.mkdir(parents=True, exist_ok=True)
    
    # Analyze
    matches = analyze_videos(cam1_path, cam2_path, args.samples)
    
    if matches:
        # Save CSV
        csv_path = output_base.parent / (output_base.stem + '.csv')
        save_csv(matches, csv_path)
        
        # Generate visualizations
        generate_visualizations(matches, output_base)
        
        # Print summary
        print_summary(matches)
    else:
        print("No matching frames found - check that both videos have visible QR codes")


if __name__ == '__main__':
    main()
