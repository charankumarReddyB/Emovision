"""
Empirical CV Performance & Latency Benchmark Suite.
Measures real-time latencies (SCRFD detection, 5-point alignment, ONNX batch inference, total pipeline latency)
and throughput (FPS) for 1, 2, 5, and 10 faces on the development machine.
"""
import sys
import time
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.services.scrfd_detector import SCRFDDetector
from app.services.face_aligner import FaceAligner
from app.services.emotion_classifier import EmotionClassifier

def run_cv_benchmark(num_warmup: int = 5, num_runs: int = 20):
    print("=" * 80)
    print("EMOVISION CV LATENCY & FPS EMPIRICAL BENCHMARK SUITE")
    print("=" * 80)

    # Initialize CV components
    detector = SCRFDDetector()
    aligner = FaceAligner(target_size=(112, 112))
    classifier = EmotionClassifier()

    # Generate synthetic camera frame (640x480)
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)

    # Warmup runs
    print("\nRunning initial component warmup...")
    for _ in range(num_warmup):
        _ = detector.detect_faces(frame)
        _ = classifier.predict_face(np.full((112, 112, 3), 100, dtype=np.uint8))

    # Benchmark SCRFD Detection
    det_latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = detector.detect_faces(frame)
        det_latencies.append((time.perf_counter() - t0) * 1000.0)
    
    avg_det_ms = np.mean(det_latencies)
    print(f"\n[1] SCRFD-500M Face Detection Latency:")
    print(f"    Average Latency : {avg_det_ms:.2f} ms")
    print(f"    Min / Max       : {np.min(det_latencies):.2f} ms / {np.max(det_latencies):.2f} ms")

    # Benchmark Alignment Latency
    sample_kps = np.array([[38, 51], [73, 51], [56, 71], [41, 92], [70, 92]], dtype=np.float32)
    sample_bbox = (100, 100, 140, 140)
    align_latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = aligner.align_face(frame, sample_kps, sample_bbox)
        align_latencies.append((time.perf_counter() - t0) * 1000.0)
    
    avg_align_ms = np.mean(align_latencies)
    print(f"\n[2] 5-Point Facial Landmark Alignment Latency:")
    print(f"    Average Latency : {avg_align_ms:.2f} ms per face")

    # Benchmark Emotion Inference & Total Pipeline across N faces (1, 2, 5, 10)
    face_counts = [1, 2, 5, 10]
    print(f"\n[3] End-to-End Pipeline Performance across N Faces:")
    print("-" * 80)
    print(f"{'Faces (N)':<10} | {'Infer Latency (ms)':<20} | {'Total Latency (ms)':<20} | {'Throughput (FPS)':<18}")
    print("-" * 80)

    sample_chip = np.full((112, 112, 3), 120, dtype=np.uint8)

    for n in face_counts:
        chips = [sample_chip.copy() for _ in range(n)]
        kps_list = [sample_kps for _ in range(n)]

        # Benchmark ONNX batch inference
        infer_latencies = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = classifier.classify_batch(chips, kps_list=kps_list)
            infer_latencies.append((time.perf_counter() - t0) * 1000.0)
        
        avg_infer_ms = np.mean(infer_latencies)
        
        # Calculate Total Latency = SCRFD + (N * Alignment) + ONNX Batch Inference
        total_lat_ms = avg_det_ms + (n * avg_align_ms) + avg_infer_ms
        fps = 1000.0 / total_lat_ms

        print(f"N = {n:<6d} | {avg_infer_ms:18.2f} ms | {total_lat_ms:18.2f} ms | {fps:16.1f} FPS")

    print("-" * 80)
    print("=" * 80)

if __name__ == "__main__":
    run_cv_benchmark()
