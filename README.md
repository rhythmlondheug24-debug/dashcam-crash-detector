# 🚗 Vehicle Proximity Detection System

![Python](https://img.shields.io/badge/Python-3.14-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11%2Bcu128-ee4c2c)
![YOLO](https://img.shields.io/badge/YOLO-26X-yellow)
![Gradio](https://img.shields.io/badge/Gradio-6.13-ff7c00)

A high-performance, GPU-accelerated video analysis pipeline designed for real-time vehicle proximity and crash detection from dashcam footage. 

Built specifically for NVIDIA RTX 50-series (Blackwell) hardware but backwards compatible with standard CUDA environments.

## ✨ Features

- **Real-Time Object Detection**: Uses the latest **YOLO26X** (Extra Large) model for maximum accuracy vehicle detection (cars, trucks, buses, motorcycles).
- **Time-to-Collision (TTC) Math**: Calculates approach speeds based on bounding box growth rates to estimate seconds until impact.
- **Smart Tracking**: Integrates **ByteTrack** with tuned configuration specifically designed to handle high-speed dashcam motion blur and reduce flickering.
- **Crash Detection**: Detects rapid approach events (>8% bounding box growth over 10 frames) and triggers dynamic visual alerts.
- **Browser-Ready Output**: Automatically transcodes processed video to H.264 using a statically linked FFmpeg binary for seamless browser playback.
- **Premium Web UI**: Features a dark-themed Gradio 6.0 interface with live progress tracking, Markdown summary reports, and a "Top Danger Frames" screenshot gallery.
- **Ownership Watermark**: Every processed video is stamped with a permanent visual watermark for intellectual property protection.

## 🛠️ Hardware Requirements

- **GPU**: NVIDIA GPU with at least 8GB VRAM (Tested on RTX 5050 Blackwell).
- **CUDA**: Environment requires CUDA 12.8 (or compatible) for FP16 inference.

## 🚀 Installation & Setup (Windows)

This repository includes an automated batch script that handles all virtual environment creation, CUDA installations, and dependency management.

1. Clone the repository:
   ```cmd
   git clone https://github.com/rhythmlondheug24-debug/dashcam-crash-detector.git
   cd dashcam-crash-detector
   ```

2. Run the automated installer:
   - Double-click `setup_and_run.bat`
   - *Note: On the very first run, it will download PyTorch (~2.5GB) and the YOLO26X model weights (`yolo26x.pt` ~130MB).*

3. The script will automatically launch the Gradio web server at `http://localhost:7860`.

## 📁 Project Structure

- `app.py`: The Gradio 6.0 web interface and CSS styling.
- `vehicle_proximity_detector.py`: The core pipeline (YOLO inference, ByteTrack, TTC math, drawing utilities, and FFmpeg conversion).
- `tracker_config.yaml`: Custom ByteTrack parameters tuned for dashcam footage.
- `setup_and_run.bat`: One-click environment bootstrap and launch script.
- `requirements.txt`: Python dependencies.

## 🧠 What is `yolo26x.pt`?

If you see a `yolo26x.pt` file in your directory, **these are the AI's "brains."** 
It stands for "PyTorch Tensor" (`.pt`). It contains the pre-trained weights (the mathematical matrices) that teach the YOLO26 algorithm what a car, truck, or motorcycle looks like. YOLO26 (released January 2026) is the latest generation with native NMS-free inference for significantly lower latency. The script downloads this automatically from Ultralytics the first time you run it.

## 📝 License
**Copyright © 2026 Rhythm Londhe. All Rights Reserved.**
This software and associated documentation files are not open-source. Unauthorized copying, modification, distribution, or use of this code is strictly prohibited.
