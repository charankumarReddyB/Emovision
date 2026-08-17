"""
Face Detector Wrapper Service using OpenCV YuNet DNN / SCRFD ONNX Models.
Detects N faces (including small/background faces) with 5 facial keypoints for 112x112 geometric alignment.
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from app.core.config import settings
from app.services.scrfd_detector import SCRFDDetector
from app.services.face_aligner import FaceAligner

class FaceDetector:
    """
    Wrapper for OpenCV YuNet / SCRFD face detector and 5-point facial keypoint aligner.
    Uses score_threshold=0.20 to catch background and partially occluded faces.
    """
    def __init__(self, score_threshold: float = 0.20, nms_threshold: float = 0.30):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.aligner = FaceAligner(target_size=(112, 112))
        
        self.yunet_detector = None
        yunet_path = settings.MODELS_DIR / "face_detection_yunet_2023mar.onnx"
        if yunet_path.exists():
            try:
                self.yunet_detector = cv2.FaceDetectorYN.create(
                    str(yunet_path),
                    "",
                    (640, 480),
                    score_threshold=score_threshold,
                    nms_threshold=nms_threshold
                )
            except Exception as e:
                print(f"[FaceDetector Warning] Could not load YuNet detector: {e}")

        self.scrfd_detector = SCRFDDetector(score_threshold=score_threshold, nms_threshold=nms_threshold)

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects N faces in image frame with 5 facial keypoints and 112x112 aligned chips.
        
        Args:
            frame (np.ndarray): BGR image frame.
            
        Returns:
            List[Dict[str, Any]]: List of face detection dicts.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        results = []

        # 1. Try OpenCV YuNet Detector first
        if self.yunet_detector is not None:
            self.yunet_detector.setInputSize((w, h))
            _, faces = self.yunet_detector.detect(frame)
            if faces is not None and len(faces) > 0:
                for face in faces:
                    x, y, bw, bh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    x = max(0, x)
                    y = max(0, y)
                    bw = min(w - x, bw)
                    bh = min(h - y, bh)
                    
                    if bw < 10 or bh < 10:
                        continue
                        
                    # Extract 5 keypoints: [left_eye, right_eye, nose, left_mouth, right_mouth]
                    # YuNet outputs: right_eye(4:6), left_eye(6:8), nose(8:10), right_mouth(10:12), left_mouth(12:14)
                    r_eye = face[4:6]
                    l_eye = face[6:8]
                    nose = face[8:10]
                    r_mouth = face[10:12]
                    l_mouth = face[12:14]
                    score = float(face[14])
                    
                    # Standard order: [left_eye, right_eye, nose, left_mouth, right_mouth]
                    kps = np.array([l_eye, r_eye, nose, l_mouth, r_mouth], dtype=np.float32)
                    
                    chip = frame[y:y+bh, x:x+bw].copy() if bw > 0 and bh > 0 else np.zeros((112, 112, 3), dtype=np.uint8)
                    aligned_chip = self.aligner.align_face(frame, kps, (x, y, bw, bh))
                    
                    results.append({
                        "bbox": (x, y, bw, bh),
                        "confidence": score,
                        "kps": kps,
                        "face_chip": chip,
                        "aligned_chip": aligned_chip
                    })
                return results

        # 2. Fallback to SCRFD Detector
        scrfd_results = self.scrfd_detector.detect_faces(frame)
        for det in scrfd_results:
            bbox = det["bbox"]
            kps = det["kps"]
            aligned_chip = self.aligner.align_face(frame, kps, bbox)
            
            results.append({
                "bbox": bbox,
                "confidence": det["confidence"],
                "kps": kps,
                "face_chip": det["face_chip"],
                "aligned_chip": aligned_chip
            })
            
        return results
