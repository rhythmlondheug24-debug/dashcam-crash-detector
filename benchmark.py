import time
import torch
import cv2
from ultralytics import YOLO

def run_benchmark():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    print("Loading model...")
    model = YOLO("yolo11s.pt")
    model.to(device)
    
    print("Generating dummy frame (640x640)...")
    import numpy as np
    dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    print("Warming up model (10 frames)...")
    for _ in range(10):
        model.track(dummy_frame, persist=True, device=device, half=True, verbose=False)
        
    print("Benchmarking model (100 frames)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        model.track(dummy_frame, persist=True, device=device, half=True, verbose=False)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    fps = 100 / (t1 - t0)
    print(f"Model Inference FPS: {fps:.2f}")

if __name__ == "__main__":
    run_benchmark()
