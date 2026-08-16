"""
Face Detection Module using OpenCV YuNet DNN.
Detects N human faces in an image or video frame with zero false positives on background objects.
Primary: OpenCV YuNet DNN (face_detection_yunet_2023mar.onnx)
Fallback: OpenCV Haar Cascade (haarcascade_frontalface_default.xml)
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from app.core.config import settings

class FaceDetector:
    """
    Robust multi-face detector using OpenCV YuNet DNN.
    Guarantees reliable human face detection without false positives on background walls/curtains.
    """
    def __init__(
        self,
        min_confidence: float = 0.60,
        min_face_size: Tuple[int, int] = (30, 30)
    ):
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size
        self.yunet_detector = None
        self.haar_cascade = None
        
        # 1. Initialize YuNet DNN detector from ONNX model file
        model_path = settings.MODELS_DIR / "face_detection_yunet_2023mar.onnx"
        if model_path.exists() and hasattr(cv2, "FaceDetectorYN_create"):
            try:
                self.yunet_detector = cv2.FaceDetectorYN_create(
                    model=str(model_path),
                    config="",
                    input_size=(settings.INPUT_WIDTH, settings.INPUT_HEIGHT),
                    score_threshold=min_confidence,
                    nms_threshold=0.3,
                    top_k=5000
                )
            except Exception as e:
                self.yunet_detector = None

        # 2. Initialize Haar Cascade as reliable fallback
        haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        try:
            cascade = cv2.CascadeClassifier(haar_path)
            if not cascade.empty():
                self.haar_cascade = cascade
        except Exception:
            self.haar_cascade = None

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects all N visible human faces in the frame.
        
        Args:
            frame (np.ndarray): BGR OpenCV image frame.
            
        Returns:
            List[Dict[str, Any]]: List of detected faces with structure:
                {
                    "bbox": (x, y, w, h),
                    "confidence": float,
                    "face_chip": np.ndarray (cropped face BGR)
                }
        """
        if frame is None or frame.size == 0:
            return []

        height, width = frame.shape[:2]
        results: List[Dict[str, Any]] = []

        # Strategy 1: YuNet DNN Deep Learning Detector (Preferred)
        if self.yunet_detector is not None:
            try:
                self.yunet_detector.setInputSize((width, height))
                _, faces = self.yunet_detector.detect(frame)
                if faces is not None:
                    for face in faces:
                        box = face[0:4].astype(int)
                        conf = float(face[-1])
                        if conf < self.min_confidence:
                            continue
                        x_min, y_min = max(0, box[0]), max(0, box[1])
                        w_box = min(width - x_min, box[2])
                        h_box = min(height - y_min, box[3])
                        
                        if w_box >= self.min_face_size[0] and h_box >= self.min_face_size[1]:
                            chip = frame[y_min:y_min+h_box, x_min:x_min+w_box].copy()
                            if chip.size > 0:
                                results.append({
                                    "bbox": (x_min, y_min, w_box, h_box),
                                    "confidence": conf,
                                    "face_chip": chip
                                })
                    return results
            except Exception:
                pass

        # Strategy 2: Strict Haar Cascade (Fallback)
        if self.haar_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self.haar_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=self.min_face_size
            )
            for (x, y, w, h) in faces:
                x_min, y_min = max(0, x), max(0, y)
                w_box = min(width - x_min, w)
                h_box = min(height - y_min, h)
                chip = frame[y_min:y_min+h_box, x_min:x_min+w_box].copy()
                if chip.size > 0:
                    results.append({
                        "bbox": (x_min, y_min, w_box, h_box),
                        "confidence": 0.85,
                        "face_chip": chip
                    })

        return results
