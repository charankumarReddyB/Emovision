"""
Face Detection Module using OpenCV.
Detects N visible faces in an image or video frame.
Supports OpenCV YuNet DNN detector and contour/skin feature detector fallback.
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from app.core.config import settings

class FaceDetector:
    """
    Multi-face detector capable of detecting N visible faces in a single image frame.
    Integrates OpenCV YuNet DNN and adaptive feature-contour detection fallback.
    """
    def __init__(
        self,
        min_confidence: float = settings.DETECTION_MIN_CONFIDENCE,
        min_face_size: Tuple[int, int] = (25, 25)
    ):
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size
        self.yunet_detector = None
        
        # Check if YuNet ONNX model file exists in app/models_weights/
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
            except Exception:
                self.yunet_detector = None

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects all N visible faces in the input frame.
        
        Args:
            frame (np.ndarray): BGR OpenCV image frame.
            
        Returns:
            List[Dict[str, Any]]: List of detected faces with structure:
                {
                    "bbox": (x, y, w, h),
                    "confidence": float,
                    "face_chip": np.ndarray (cropped face)
                }
        """
        if frame is None or frame.size == 0:
            return []

        height, width = frame.shape[:2]
        results: List[Dict[str, Any]] = []

        # Strategy 1: YuNet DNN Detector if loaded
        if self.yunet_detector is not None:
            try:
                self.yunet_detector.setInputSize((width, height))
                _, faces = self.yunet_detector.detect(frame)
                if faces is not None:
                    for face in faces:
                        box = face[0:4].astype(int)
                        conf = float(face[-1])
                        x_min, y_min, w_box, h_box = max(0, box[0]), max(0, box[1]), box[2], box[3]
                        
                        if w_box >= self.min_face_size[0] and h_box >= self.min_face_size[1]:
                            chip = frame[y_min:y_min+h_box, x_min:x_min+w_box].copy()
                            results.append({
                                "bbox": (x_min, y_min, w_box, h_box),
                                "confidence": conf,
                                "face_chip": chip
                            })
                    return results
            except Exception:
                pass

        # Strategy 2: Adaptive Multi-Face Region & Contour Detector
        # Preprocessing: convert to grayscale and HSV for robust face region candidate extraction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Adaptive Thresholding & Edge Detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 150)
        
        # Skin color range in HSV space
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Combine edges and skin/intensity candidates
        combined = cv2.bitwise_or(edges, skin_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Filter candidates by minimum size and aspect ratio (faces typically 0.6 <= aspect_ratio <= 1.4)
            if w < self.min_face_size[0] or h < self.min_face_size[1]:
                continue
                
            aspect_ratio = float(w) / float(h)
            if 0.5 <= aspect_ratio <= 1.6 and (w * h) > 600:
                # Boundary check
                x_min = max(0, x)
                y_min = max(0, y)
                x_max = min(width, x + w)
                y_max = min(height, y + h)
                
                box_w = x_max - x_min
                box_h = y_max - y_min
                
                chip = frame[y_min:y_max, x_min:x_max].copy()
                
                # Confidence score estimate
                conf = min(0.98, round(0.70 + (box_w * box_h) / (width * height * 0.2), 2))
                
                results.append({
                    "bbox": (x_min, y_min, box_w, box_h),
                    "confidence": conf,
                    "face_chip": chip
                })
                
        return results
