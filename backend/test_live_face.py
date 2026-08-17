"""
Direct Live Webcam Face Verification & Preprocessing Debug Script for POSTER FER Model.

Tests:
1. Webcam capture via OpenCV.
2. SCRFD face detection & keypoint extraction.
3. Aligned face crop generation + debug image saving (debug_face.png, debug_face_raw.png).
4. POSTER 71.85M batch model forward pass.
5. All 7 emotion probabilities printout & latency measurement.
"""
import sys
import time
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.services.scrfd_detector import SCRFDDetector
from app.services.face_aligner import FaceAligner
from app.services.emotion_classifier import EmotionClassifier, POSTER_CLASSES

def run_live_webcam_test():
    print("=" * 80)
    print("EMOVISION LIVE WEBCAM & POSTER PREPROCESSING VERIFICATION TEST")
    print("=" * 80)

    # 1. Hardware Check
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Only (Intel/AMD)"
    print(f"\n[HARDWARE ACCELERATION AUDIT]")
    print(f"  • CUDA Available : {cuda_avail}")
    print(f"  • Device Name    : {device_name}")
    print(f"  • PyTorch Thread : {torch.get_num_threads()} CPU threads")

    # 2. Initialize Services
    print(f"\n[INITIALIZING COMPUTER VISION PIPELINE]")
    detector = SCRFDDetector(score_threshold=0.35, nms_threshold=0.30)
    aligner = FaceAligner(target_size=(224, 224))
    classifier = EmotionClassifier()

    if not classifier.is_weights_loaded:
        print("ERROR: POSTER model weights failed to load!")
        return

    # 3. Open Webcam
    print(f"\n[WEBCAM FEED ACCESS]")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("WARNING: Could not open default webcam index 0.")
        print("Testing pipeline on static sample face from RAF-DB test set...")
        sample_path = Path(__file__).resolve().parent / "data" / "rafdb_test" / "DATASET" / "test" / "4" / "test_0003_aligned.jpg"
        if sample_path.exists():
            frame = cv2.imread(str(sample_path))
        else:
            frame = np.full((480, 640, 3), 120, dtype=np.uint8)
            cv2.rectangle(frame, (200, 100), (440, 380), (200, 200, 200), -1)
    else:
        print("Webcam initialized successfully! Capturing live test frame...")
        # Warmup camera
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            print("ERROR: Failed to read frame from webcam.")
            return

    h, w = frame.shape[:2]
    print(f"  • Live Frame Size : {w}x{h}")

    # 4. SCRFD Multi-Face Detection
    t_start = time.time()
    detections = detector.detect_faces(frame)
    t_det = time.time() - t_start

    print(f"\n[SCRFD MULTI-FACE DETECTION RESULT]")
    print(f"  • Face Count (N)  : {len(detections)}")
    print(f"  • Detection Time  : {t_det * 1000.0:.2f} ms")

    if not detections:
        print("No faces detected in the current webcam frame.")
        return

    # Process all detected faces
    face_chips = []
    aligned_chips = []
    
    t_align_start = time.time()
    for idx, det in enumerate(detections):
        bbox = det["bbox"]
        kps = det["kps"]
        
        # Raw bbox crop
        x, y, bw, bh = bbox
        raw_crop = frame[y:y+bh, x:x+bw].copy() if bw > 0 and bh > 0 else np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Aligned 5-point face crop
        aligned_crop = aligner.align_face(frame, kps, bbox)
        
        face_chips.append(raw_crop)
        aligned_chips.append(aligned_crop)

        # Save debug images for Face 1
        if idx == 0:
            debug_face_path = Path(__file__).resolve().parent / "debug_face.png"
            debug_raw_path = Path(__file__).resolve().parent / "debug_face_raw.png"
            
            cv2.imwrite(str(debug_face_path), aligned_crop)
            cv2.imwrite(str(debug_raw_path), raw_crop)
            
            print(f"\n[DEBUG IMAGE SAVED]")
            print(f"  • Aligned Face Chip : {debug_face_path.resolve()}")
            print(f"  • Raw Face BBox     : {debug_raw_path.resolve()}")

    t_align = time.time() - t_align_start

    # 5. POSTER Batch Inference
    t_infer_start = time.time()
    
    # Preprocess all faces into single batch tensor [N, 3, 224, 224]
    tensors = [classifier.transform(chip) for chip in aligned_chips]
    batch_tensor = torch.stack(tensors).to(classifier.device)
    
    with torch.inference_mode():
        outputs, _ = classifier.poster_model(batch_tensor)
        probs_batch = F.softmax(outputs, dim=1).cpu().numpy()

    t_infer = time.time() - t_infer_start
    t_total = t_det + t_align + t_infer
    fps = 1.0 / t_total if t_total > 0 else 0.0

    # 6. Detailed Per-Face Report
    print("\n" + "=" * 80)
    print("LIVE WEBCAM INFERENCE DETAILED DEBUG REPORT")
    print("=" * 80)
    print(f"FACE COUNT: {len(detections)}")

    for idx, (det, probs) in enumerate(zip(detections, probs_batch)):
        top_idx = int(np.argmax(probs))
        conf = float(probs[top_idx])
        label = "Uncertain" if conf < 0.40 else POSTER_CLASSES[top_idx]

        print(f"\nFACE {idx + 1}:")
        print(f"  • BBox           : {det['bbox']} (Width: {det['bbox'][2]}, Height: {det['bbox'][3]})")
        print(f"  • Detection Conf : {det['confidence'] * 100:.1f}%")
        print(f"  • Prediction     : {label}")
        print(f"  • Confidence     : {conf * 100:.1f}%")
        print(f"  • All 7 Probabilities:")
        for class_idx, class_name in enumerate(POSTER_CLASSES):
            print(f"      {class_name:10s} : {probs[class_idx] * 100:6.2f}%")

    print("\n" + "-" * 80)
    print(f"LATENCY & PERFORMANCE METRICS:")
    print(f"  • SCRFD Detection Latency : {t_det * 1000.0:6.2f} ms")
    print(f"  • Alignment Latency       : {t_align * 1000.0:6.2f} ms")
    print(f"  • POSTER Batch Latency    : {t_infer * 1000.0:6.2f} ms (Batch Size: {len(detections)})")
    print(f"  • Total Pipeline Latency  : {t_total * 1000.0:6.2f} ms")
    print(f"  • Effective Pipeline FPS  : {fps:6.2f} FPS")
    print("=" * 80)

if __name__ == "__main__":
    run_live_webcam_test()
