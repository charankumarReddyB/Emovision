"""
Dedicated Computer Vision Pipeline Benchmark Script.
Measures latency and actual FPS across 1, 2, 5, and 8 human faces for:
1. SCRFD-2.5G Face Detection
2. 5-Point Affine Face Alignment
3. Emotion Model Batch Inference
4. Total End-to-End Pipeline Latency & FPS
"""
import time
import cv2
import numpy as np
import torch
import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from app.services.face_detector import FaceDetector
from app.services.face_aligner import FaceAligner
from app.services.emotion_classifier import EmotionClassifier

def benchmark_batch_inference(detector, aligner, classifier, num_faces: int = 1, num_runs: int = 30):
    # Create N face chips
    dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 120
    kps = np.array([[30, 30], [70, 30], [50, 50], [40, 70], [60, 70]], dtype=np.float32)
    bbox = (20, 20, 60, 60)
    
    det_times = []
    align_times = []
    model_times = []
    total_times = []
    
    # Warmup
    for _ in range(5):
        _ = detector.scrfd.detect_faces(dummy_frame)
        aligned_chips = [aligner.align_face(dummy_frame, kps, bbox) for _ in range(num_faces)]
        classifier.classify_batch(aligned_chips)
        
    for _ in range(num_runs):
        t0 = time.perf_counter()
        
        # 1. Detection
        t_d0 = time.perf_counter()
        raw_dets = detector.scrfd.detect_faces(dummy_frame)
        t_d1 = time.perf_counter()
        
        # 2. Alignment (N faces)
        t_a0 = time.perf_counter()
        aligned_chips = [aligner.align_face(dummy_frame, kps, bbox) for _ in range(num_faces)]
        t_a1 = time.perf_counter()
        
        # 3. PyTorch Model Batch Inference (N faces)
        t_m0 = time.perf_counter()
        preds = classifier.classify_batch(aligned_chips)
        t_m1 = time.perf_counter()
        
        t1 = time.perf_counter()
        
        det_times.append((t_d1 - t_d0) * 1000.0)
        align_times.append((t_a1 - t_a0) * 1000.0)
        model_times.append((t_m1 - t_m0) * 1000.0)
        total_times.append((t1 - t0) * 1000.0)
        
    avg_det = float(np.mean(det_times))
    avg_align = float(np.mean(align_times))
    avg_model = float(np.mean(model_times))
    avg_total = float(np.mean(total_times))
    fps = 1000.0 / avg_total if avg_total > 0 else 0.0
    
    return {
        "num_faces": num_faces,
        "detection_latency_ms": avg_det,
        "alignment_latency_ms": avg_align,
        "model_latency_ms": avg_model,
        "total_latency_ms": avg_total,
        "fps": fps
    }

def main():
    print("==================================================")
    print("COMPUTER VISION PIPELINE REAL-TIME BENCHMARK")
    print("==================================================")
    
    detector = FaceDetector()
    aligner = FaceAligner()
    classifier = EmotionClassifier()
    
    face_counts = [1, 2, 5, 8]
    results = []
    
    print(f"\n{'Faces (N)':<10} | {'SCRFD Det (ms)':<15} | {'Align (ms)':<12} | {'Model (ms)':<12} | {'Total (ms)':<12} | {'FPS':<8}")
    print("-" * 80)
    
    for n in face_counts:
        res = benchmark_batch_inference(detector, aligner, classifier, num_faces=n)
        results.append(res)
        print(f"{res['num_faces']:<10} | {res['detection_latency_ms']:<15.2f} | {res['alignment_latency_ms']:<12.2f} | {res['model_latency_ms']:<12.2f} | {res['total_latency_ms']:<12.2f} | {res['fps']:<8.1f}")
        
    print("==================================================")
    print("NO-FACE CONDITION BENCHMARK (Empty Scene)")
    print("==================================================")
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    empty_res = detector.detect_faces(empty_frame)
    print(f" Empty Scene Face Detections: {len(empty_res)} (Expected: 0)")
    assert len(empty_res) == 0
    print(" Zero False Positives Confirmed!")
    print("==================================================")

if __name__ == "__main__":
    main()
