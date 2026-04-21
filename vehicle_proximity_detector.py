"""
Vehicle Proximity Detection System v2
=======================================
Production-ready dashcam analysis pipeline using YOLOv11s + ByteTrack.
Detects vehicles, tracks them, estimates time-to-collision, detects
crash moments, and triggers dynamic visual alerts.

Hardware Target: NVIDIA RTX 5050 (8GB VRAM), 24GB System RAM
Inference: FP16 CUDA-only | Tracking: ByteTrack

v2 Improvements:
  - Upgraded model: yolo11s (small) for better accuracy
  - Time-to-Collision (TTC) estimation per vehicle
  - Crash moment detection via bbox growth rate analysis
  - Auto-export top danger frame screenshots
  - H.264 output codec for browser/Gradio compatibility
  - Post-processing summary with timestamps
"""

import sys
import gc
import os
import time
import subprocess
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import deque
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Proximity thresholds
THREAT_THRESHOLD_DEFAULT = 0.07   # 7% of frame = danger
THREAT_THRESHOLD_CENTER = 0.04    # 4% if vehicle is in center lane
CENTER_LANE_LEFT = 0.30
CENTER_LANE_RIGHT = 0.70

# Visual constants
COLOR_SAFE = (0, 255, 0)
COLOR_THREAT = (0, 0, 255)
COLOR_TTC_WARN = (0, 180, 255)    # Orange for TTC warnings
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICKNESS = 1
BORDER_THICKNESS = 10

# Model config — yolo11s for better accuracy on 8GB VRAM
MODEL_NAME = "yolo11s.pt"
CONFIDENCE_THRESHOLD = 0.20
TRACKER_CONFIG = "tracker_config.yaml"

# Crash detection
CRASH_GROWTH_WINDOW = 10          # Frames to measure bbox growth over
CRASH_GROWTH_THRESHOLD = 0.08    # 8% ratio growth in window = rapid approach
TRACK_HISTORY_LENGTH = 30         # Frames of history per track
MAX_DANGER_SCREENSHOTS = 5        # Top N danger frames to save


# ═══════════════════════════════════════════════════════════════════════════
# CUDA VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def verify_cuda():
    """Verify CUDA is available and print GPU info. Halts if CPU-only."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. This app requires an NVIDIA GPU.\n"
            "Install PyTorch with CUDA:\n"
            "  pip install torch torchvision torchaudio --index-url "
            "https://download.pytorch.org/whl/cu128"
        )
    device = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024 ** 3)
    print(f"[GPU] {device} | VRAM: {vram:.1f} GB")
    print(f"[CUDA] {torch.version.cuda} | [PyTorch] {torch.__version__}")
    return "cuda:0"


# ═══════════════════════════════════════════════════════════════════════════
# PROXIMITY MATH
# ═══════════════════════════════════════════════════════════════════════════

def calculate_threat(x1, y1, x2, y2, frame_w, frame_h):
    """
    Determine if a vehicle is a proximity threat using 2D bbox geometry.
    Center vector weighting: stricter threshold in the middle 40% of X-axis.

    Returns: (is_threat, ratio, threshold_used)
    """
    box_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_w * frame_h
    ratio = box_area / frame_area if frame_area > 0 else 0.0

    cx_norm = ((x1 + x2) / 2.0) / frame_w
    if CENTER_LANE_LEFT <= cx_norm <= CENTER_LANE_RIGHT:
        threshold = THREAT_THRESHOLD_CENTER
    else:
        threshold = THREAT_THRESHOLD_DEFAULT

    return ratio > threshold, ratio, threshold


# ═══════════════════════════════════════════════════════════════════════════
# TRACK ANALYZER — Crash Detection + TTC Estimation
# ═══════════════════════════════════════════════════════════════════════════

class TrackAnalyzer:
    """Tracks per-vehicle bbox history to detect crashes and estimate TTC."""

    def __init__(self, fps):
        self.fps = fps
        self.history = {}               # track_id -> deque of (frame, ratio)
        self.crash_moments = []         # Detected rapid approach events
        self.max_threat_ratio = 0.0
        self.total_threat_frames = 0
        self.threat_frame_set = set()   # Unique frames with threats

    def update(self, track_id, frame_num, ratio, is_threat):
        """Record a detection and check for crash-like approach patterns."""
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=TRACK_HISTORY_LENGTH)
        self.history[track_id].append((frame_num, ratio))

        if is_threat:
            self.total_threat_frames += 1
            self.threat_frame_set.add(frame_num)
            self.max_threat_ratio = max(self.max_threat_ratio, ratio)

        # Detect crash moment: rapid bbox growth over N frames
        hist = self.history[track_id]
        if len(hist) >= CRASH_GROWTH_WINDOW:
            old_ratio = hist[-CRASH_GROWTH_WINDOW][1]
            growth = ratio - old_ratio
            if growth > CRASH_GROWTH_THRESHOLD:
                # Avoid duplicate crash events within 2-second windows
                ts = frame_num / self.fps
                if not self.crash_moments or (ts - self.crash_moments[-1]['timestamp'] > 2.0):
                    self.crash_moments.append({
                        'frame': frame_num,
                        'track_id': int(track_id),
                        'ratio': ratio,
                        'timestamp': ts,
                        'growth_rate': growth,
                    })

    def get_ttc(self, track_id):
        """Estimate time-to-collision in seconds based on bbox growth rate."""
        if track_id not in self.history:
            return None
        hist = self.history[track_id]
        if len(hist) < 5:
            return None

        current_ratio = hist[-1][1]
        past_ratio = hist[-5][1]
        growth_per_frame = (current_ratio - past_ratio) / 5.0

        if growth_per_frame <= 0.001:
            return None  # Not approaching

        frames_left = (1.0 - current_ratio) / growth_per_frame
        ttc = frames_left / self.fps

        if ttc < 0 or ttc > 30:
            return None  # Unreasonable range
        return round(ttc, 1)

    def generate_summary(self, total_frames):
        """Generate a markdown summary of the analysis."""
        duration = total_frames / self.fps if self.fps > 0 else 0
        lines = [
            "## 📊 Analysis Summary\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Frames | {total_frames} |",
            f"| Duration | {duration:.1f}s |",
            f"| Vehicles Tracked | {len(self.history)} |",
            f"| Frames with Threats | {len(self.threat_frame_set)} |",
            f"| Peak Threat Ratio | {self.max_threat_ratio:.1%} |",
            "",
        ]

        if self.crash_moments:
            lines.append("### ⚠️ Rapid Approach / Crash Moments\n")
            for i, cm in enumerate(self.crash_moments[:10]):
                mins = int(cm['timestamp'] // 60)
                secs = cm['timestamp'] % 60
                lines.append(
                    f"**{i+1}.** `{mins:02d}:{secs:05.2f}` — "
                    f"Vehicle #{cm['track_id']} at **{cm['ratio']:.0%}** "
                    f"screen coverage (growth: +{cm['growth_rate']:.0%})"
                )
            lines.append("")
        else:
            lines.append("### ✅ No Rapid Approach Events Detected\n")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# DRAWING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def draw_text_with_bg(frame, text, org, font_scale, color, bg_color, alpha=0.6):
    """Draw text with a semi-transparent background rectangle."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, FONT_THICKNESS)
    x, y = org
    pad = 4
    rx1, ry1 = x - pad, y - th - pad - baseline
    rx2, ry2 = x + tw + pad, y + pad

    h, w = frame.shape[:2]
    rx1, ry1 = max(0, rx1), max(0, ry1)
    rx2, ry2 = min(w, rx2), min(h, ry2)

    if rx2 <= rx1 or ry2 <= ry1:
        return

    overlay = frame[ry1:ry2, rx1:rx2].copy()
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), bg_color, -1)
    cv2.addWeighted(frame[ry1:ry2, rx1:rx2], alpha, overlay, 1 - alpha, 0,
                    frame[ry1:ry2, rx1:rx2])
    cv2.putText(frame, text, (x, y), FONT, font_scale, color, FONT_THICKNESS,
                cv2.LINE_AA)


def draw_global_alarm(frame):
    """Draw a 10px solid red border around the entire frame."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_THREAT, BORDER_THICKNESS)


def annotate_vehicle(frame, x1, y1, x2, y2, track_id, is_threat, ratio, ttc=None):
    """Draw bounding box, label, and optional TTC for a single vehicle."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    if is_threat:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_THREAT, 5)
        label = f"WARNING: PROXIMITY [{track_id}] {ratio:.0%}"
        if ttc is not None:
            label += f" | TTC: {ttc}s"
        draw_text_with_bg(frame, label, (x1, y1 - 6), FONT_SCALE,
                          COLOR_WHITE, COLOR_THREAT, alpha=0.85)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_SAFE, 2)
        label = f"ID: {track_id}"
        # Show TTC in orange if vehicle is approaching (even if not yet a threat)
        if ttc is not None and ttc < 8.0:
            label += f" | TTC: {ttc}s"
            draw_text_with_bg(frame, label, (x1, y1 - 6), FONT_SCALE,
                              COLOR_WHITE, COLOR_TTC_WARN, alpha=0.7)
        else:
            draw_text_with_bg(frame, label, (x1, y1 - 6), FONT_SCALE,
                              COLOR_WHITE, COLOR_BLACK, alpha=0.6)


def draw_hud(frame, frame_num, total_frames, fps, threats_count, vehicles_count,
             min_ttc=None):
    """Draw a heads-up display with stats in the top-left corner."""
    lines = [
        f"Frame: {frame_num}/{total_frames}",
        f"FPS: {fps:.1f}",
        f"Vehicles: {vehicles_count} | Threats: {threats_count}",
    ]
    if min_ttc is not None:
        lines.append(f"Min TTC: {min_ttc}s")

    y_offset = 28
    for i, line in enumerate(lines):
        draw_text_with_bg(frame, line, (12, y_offset + i * 24), 0.5,
                          COLOR_WHITE, COLOR_BLACK, alpha=0.5)


# ═══════════════════════════════════════════════════════════════════════════
# H.264 CODEC CONVERSION
# ═══════════════════════════════════════════════════════════════════════════

def convert_to_h264(video_path, original_path=None):
    """
    Convert an mp4v video to H.264 for browser/Gradio playback.
    Uses ffmpeg from imageio-ffmpeg (ships a static binary).
    Falls back gracefully if conversion fails.
    """
    video_path = Path(video_path)
    h264_path = video_path.with_stem(video_path.stem + "_h264")

    # Find ffmpeg binary
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg_exe = "ffmpeg"  # Try system ffmpeg

    try:
        print(f"[CODEC] Converting to H.264 and merging audio...")
        
        ffmpeg_args = [
            ffmpeg_exe,
            '-y',                       # Overwrite output
            '-i', str(video_path),      # Input 0: Processed video without audio
        ]

        if original_path:
            ffmpeg_args.extend(['-i', str(original_path)])  # Input 1: Original video with audio
            
        ffmpeg_args.extend([
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', # Fix odd resolutions
            '-c:v', 'libx264',          # H.264 codec
            '-preset', 'fast',          # Speed/quality tradeoff
            '-crf', '23',               # Quality factor
            '-pix_fmt', 'yuv420p',      # Maximum compatibility
            '-movflags', '+faststart',  # Allow browser to stream before full download
        ])

        if original_path:
            ffmpeg_args.extend([
                '-c:a', 'aac',          # Audio codec
                '-map', '0:v:0',        # Take video from input 0
                '-map', '1:a:0?'        # Take audio from input 1 (if it exists)
            ])

        ffmpeg_args.append(str(h264_path))

        result = subprocess.run(ffmpeg_args, check=True, capture_output=True, timeout=600, text=True)

        # Replace original with H.264 version
        video_path.unlink()
        h264_path.rename(video_path)
        print(f"[CODEC] ✓ Converted to H.264: {video_path.name}")
        return str(video_path), None

    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg failed with exit code {e.returncode}. Stderr: {e.stderr}"
        print(f"[WARN] {error_msg}")
        h264_path.unlink(missing_ok=True)
        return str(video_path), error_msg
    except Exception as e:
        error_msg = f"H.264 conversion failed: {e}"
        print(f"[WARN] {error_msg}")
        h264_path.unlink(missing_ok=True)
        return str(video_path), error_msg


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def process_video(input_path: str, output_path: str = None, progress_callback=None):
    """
    Main processing pipeline.

    Args:
        input_path:  Path to source dashcam video (.MOV or .MP4).
        output_path: Path for output annotated video. Auto-generated if None.
        progress_callback: Optional callable(fraction, desc) for UI progress.

    Returns:
        dict with keys: output_path, summary, screenshots
    """
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_tracked.mp4"
    else:
        output_path = Path(output_path).resolve()

    # Directory for danger frame screenshots
    screenshots_dir = output_path.parent / f"{output_path.stem}_screenshots"
    screenshots_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  VEHICLE PROXIMITY DETECTION SYSTEM v2")
    print("=" * 60)

    # ── Phase 1: CUDA + Model Init ──────────────────────────────────────
    device = verify_cuda()

    print(f"\n[MODEL] Loading {MODEL_NAME} on {device} with FP16...")
    model = YOLO(MODEL_NAME)
    model.to(device)
    print(f"[MODEL] Device: {next(model.model.parameters()).device}")

    # ── Phase 2: Video Ingestion ────────────────────────────────────────
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n[VIDEO] {input_path.name} | {width}x{height} @ {fps:.2f} FPS")
    print(f"[VIDEO] {total_frames} frames | {total_frames / fps if fps > 0 else 0:.1f}s duration")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise IOError(f"Cannot initialize VideoWriter at: {output_path}")

    print(f"[OUTPUT] {output_path}\n")

    # ── Initialize Track Analyzer ───────────────────────────────────────
    analyzer = TrackAnalyzer(fps)

    # Top danger frames: list of (annotated_frame_copy, frame_num, max_ratio)
    danger_frames = []

    # ── Processing Loop ─────────────────────────────────────────────────
    frame_num = 0
    processing_fps = 0.0

    script_dir = Path(__file__).parent.resolve()
    tracker_cfg_path = script_dir / TRACKER_CONFIG
    tracker_arg = str(tracker_cfg_path) if tracker_cfg_path.exists() else "bytetrack.yaml"

    pbar = tqdm(total=total_frames, desc="Processing", unit="frame",
                bar_format="{l_bar}{bar:30}{r_bar}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            t_start = time.perf_counter()

            # ── YOLO Inference + ByteTrack ──────────────────────────────
            results = model.track(
                source=frame,
                persist=True,
                tracker=tracker_arg,
                classes=VEHICLE_CLASSES,
                conf=CONFIDENCE_THRESHOLD,
                device=device,
                half=True,
                verbose=False,
                imgsz=2560,  # Maxed out resolution for perfect detection of static/distant cars
            )

            # ── Process Detections ──────────────────────────────────────
            any_threat = False
            vehicles_count = 0
            threats_count = 0
            min_ttc = None
            max_ratio_this_frame = 0.0

            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()

                if boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = np.arange(len(xyxy))

                vehicles_count = len(xyxy)

                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    tid = track_ids[i]

                    # Proximity calculation
                    is_threat, ratio, thresh = calculate_threat(
                        x1, y1, x2, y2, width, height
                    )

                    # Feed the track analyzer
                    analyzer.update(tid, frame_num, ratio, is_threat)
                    ttc = analyzer.get_ttc(tid)

                    if is_threat:
                        any_threat = True
                        threats_count += 1
                    max_ratio_this_frame = max(max_ratio_this_frame, ratio)

                    # Track minimum TTC across all vehicles
                    if ttc is not None:
                        if min_ttc is None or ttc < min_ttc:
                            min_ttc = ttc

                    # Annotate
                    annotate_vehicle(frame, x1, y1, x2, y2, tid,
                                     is_threat, ratio, ttc)

            # ── Global Alarm ────────────────────────────────────────────
            if any_threat:
                draw_global_alarm(frame)

            # ── HUD ─────────────────────────────────────────────────────
            t_elapsed = time.perf_counter() - t_start
            processing_fps = 1.0 / t_elapsed if t_elapsed > 0 else 0
            draw_hud(frame, frame_num, total_frames, processing_fps,
                     threats_count, vehicles_count, min_ttc)

            # ── Capture Danger Frames ───────────────────────────────────
            if any_threat and max_ratio_this_frame > 0.10:
                if (len(danger_frames) < MAX_DANGER_SCREENSHOTS or
                        max_ratio_this_frame > danger_frames[-1][2]):
                    danger_frames.append(
                        (frame.copy(), frame_num, max_ratio_this_frame)
                    )
                    danger_frames.sort(key=lambda x: x[2], reverse=True)
                    if len(danger_frames) > MAX_DANGER_SCREENSHOTS:
                        danger_frames.pop()

            # ── Watermark ───────────────────────────────────────────────
            cv2.putText(frame, 'PROCESSED BY RHYTHM LONDHE', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # ── Write Frame ─────────────────────────────────────────────
            out.write(frame)

            # ── Memory Management ───────────────────────────────────────
            del results
            if frame_num % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            pbar.update(1)
            pbar.set_postfix({
                "fps": f"{processing_fps:.1f}",
                "veh": vehicles_count,
                "threats": threats_count,
                "ttc": f"{min_ttc}s" if min_ttc else "-",
            })

            if progress_callback is not None:
                progress_callback(
                    frame_num / total_frames if total_frames > 0 else 0,
                    f"Frame {frame_num}/{total_frames} | "
                    f"{processing_fps:.1f} FPS | {threats_count} threats"
                    + (f" | TTC: {min_ttc}s" if min_ttc else "")
                )

    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] User cancelled. Saving progress...")

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        raise

    finally:
        pbar.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        gc.collect()
        torch.cuda.empty_cache()

        print(f"\n{'═' * 60}")
        print(f"  PROCESSING COMPLETE — {frame_num}/{total_frames} frames")
        print(f"{'═' * 60}")

    # ── Post-Processing ─────────────────────────────────────────────────

    # Save danger frame screenshots
    screenshot_paths = []
    for img, fnum, ratio in danger_frames:
        fname = screenshots_dir / f"danger_frame_{fnum}_ratio_{int(ratio * 100)}pct.jpg"
        cv2.imwrite(str(fname), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        screenshot_paths.append(str(fname))
        print(f"[SCREENSHOT] Saved: {fname.name}")

    # Convert to H.264 and merge original audio
    final_path, codec_error = convert_to_h264(str(output_path), original_path=input_path)

    # Generate summary
    summary = analyzer.generate_summary(frame_num)
    
    if codec_error:
        summary += f"\n### ⚠️ Video Conversion Error\n"
        summary += f"The video processed perfectly, but converting it to a browser-playable format failed. You can still download the raw `.mp4` using the button in the top right corner of the black box.\n\n"
        summary += f"**Technical Details:**\n```\n{codec_error}\n```\n"

    if screenshot_paths:
        summary += f"\n### 📸 {len(screenshot_paths)} Danger Screenshots Saved\n"
        summary += f"Location: `{screenshots_dir}`\n"

    print(f"\n[DONE] Output: {final_path}")

    return {
        "output_path": final_path,
        "summary": summary,
        "screenshots": screenshot_paths,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT (CLI)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_input = sys.argv[1]
    else:
        script_dir = Path(__file__).parent.resolve()
        candidates = list(script_dir.glob("*.MOV")) + list(script_dir.glob("*.mp4"))
        if candidates:
            video_input = str(candidates[0])
            print(f"[AUTO] Found video: {candidates[0].name}")
        else:
            print("Usage: python vehicle_proximity_detector.py <path_to_video>")
            sys.exit(1)

    video_output = sys.argv[2] if len(sys.argv) > 2 else None
    result = process_video(video_input, video_output)
    print("\n" + result["summary"])
