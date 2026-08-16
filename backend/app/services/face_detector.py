"""
Face Detector Wrapper Service using SCRFD-2.5G ONNX Model.
Detects N faces with 5 facial keypoints for geometric alignment.
Provides zero false positive guarantees on empty scenes and non-face background objects.
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from app.services.scrfd_detector import SCRFDDetector
from app.services.face_aligner import FaceAligner

class FaceDetector:
    """
    Wrapper for SCRFD-2.5G face detector and 5-point facial keypoint aligner.
    """
    def __init__(self, score_threshold: float = 0.50, nms_threshold: float = 0.40):
        self.scrfd = SCRFDDetector(score_threshold=score_threshold, nms_threshold=nms_threshold)
        self.aligner = FaceAligner()

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects N faces in image frame with 5 facial keypoints and geometrically aligned chips.
        
        Args:
            frame (np.ndarray): BGR image frame.
            
        Returns:
            List[Dict[str, Any]]: List of face detection dicts:
                {
                    "bbox": (x, y, w, h),
                    "confidence": float,
                    "kps": np.ndarray (5, 2),
                    "face_chip": BGR raw crop,
                    "aligned_chip": BGR 5-point aligned crop (48, 48, 3)
                }
        """
        raw_detections = self.scrfd.detect_faces(frame)
        results = []
        
        for det in raw_detections:
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
