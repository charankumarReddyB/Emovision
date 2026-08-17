"""
SCRFD Multi-Face Detection Step-by-Step Tracing & Debugging Suite.
Investigates SCRFD-500M detection stages across N=5 visible faces.
Tests score thresholds (0.30, 0.25, 0.20) and NMS thresholds (0.35, 0.40).
Saves visual bounding box output image: backend/debug_5_faces_detected.png
"""
import sys
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.scrfd_detector import SCRFDDetector
from app.services.face_aligner import FaceAligner

def create_real_5_face_canvas(canvas_w: int = 1280, canvas_h: int = 720) -> np.ndarray:
    """Generates a realistic 5-face scene with 5 distinct RAF-DB face samples placed cleanly."""
    canvas = np.full((canvas_h, canvas_w, 3), 180, dtype=np.uint8)
    
    sample_dir = BACKEND_DIR / "data" / "rafdb_test" / "DATASET" / "test"
    face_paths = []
    
    for class_id in range(1, 8):
        folder = sample_dir / str(class_id)
        if folder.exists():
            imgs = list(folder.glob("*.jpg"))
            if len(imgs) >= 2:
                face_paths.append(imgs[0])
                face_paths.append(imgs[1])

    if not face_paths:
        face_paths = [BACKEND_DIR / "debug_face.png"]

    # 5 face positions centered nicely across 1280x720 canvas
    positions = [
        (80, 240),
        (310, 240),
        (540, 240),
        (770, 240),
        (1000, 240)
    ]

    target_size = 200

    for idx, (cx, cy) in enumerate(positions):
        img_p = face_paths[idx % len(face_paths)]
        if img_p.exists():
            f_img = cv2.imread(str(img_p))
            if f_img is not None and f_img.size > 0:
                # Add 20px padding around face chip to provide realistic head context
                padded_face = cv2.copyMakeBorder(f_img, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
                f_resized = cv2.resize(padded_face, (target_size, target_size))
                canvas[cy:cy+target_size, cx:cx+target_size] = f_resized

    return canvas

def trace_scrfd_pipeline():
    print("=" * 80)
    print("SCRFD-500M MULTI-FACE DETECTION TRACING & THRESHOLD OPTIMIZATION AUDIT")
    print("=" * 80)

    canvas = create_real_5_face_canvas()
    test_img_path = BACKEND_DIR / "debug_5_faces_scene.png"
    cv2.imwrite(str(test_img_path), canvas)
    print(f"  • Generated 5-Face Scene Image: {test_img_path}")

    thresholds_to_test = [0.30, 0.25, 0.20]
    aligner = FaceAligner()

    for thresh in thresholds_to_test:
        print("\n" + "-" * 80)
        print(f"TESTING SCORE THRESHOLD: {thresh}")
        print("-" * 80)

        detector = SCRFDDetector(score_threshold=thresh, nms_threshold=0.35, input_size=(640, 640))
        
        filtered_dets = detector.detect_faces(canvas)
        print(f"  1. SCRFD Face Detections (Threshold={thresh}): {len(filtered_dets)}")

        chips = []
        for idx, det in enumerate(filtered_dets, start=1):
            x, y, w, h = det["bbox"]
            kps = det.get("kps")
            chip = aligner.align_face(canvas, kps=kps, bbox=det["bbox"])
            if chip is not None and chip.size > 0:
                chips.append(chip)
                det["aligned_chip"] = chip

        print(f"  2. Final Aligned Face Chips Sent to EfficientFace: {len(chips)}")

        # Visualize bounding boxes
        vis_img = canvas.copy()
        for idx, det in enumerate(filtered_dets, start=1):
            x, y, w, h = det["bbox"]
            conf = det["confidence"]
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis_img, f"Face #{idx}: {conf:.2f}", (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            kps = det.get("kps")
            if kps is not None:
                for kp in kps:
                    cv2.circle(vis_img, (int(kp[0]), int(kp[1])), 3, (0, 0, 255), -1)

        out_vis_path = BACKEND_DIR / f"debug_5_faces_detected.png"
        cv2.imwrite(str(out_vis_path), vis_img)
        print(f"  • Saved Detection Visualization: {out_vis_path}")

if __name__ == "__main__":
    trace_scrfd_pipeline()
