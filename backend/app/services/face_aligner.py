"""
Face Aligner Service for Emovision CV Core.
Uses 5 facial keypoints (left_eye, right_eye, nose, left_mouth, right_mouth) to perform
geometric 2D affine transformation warping before emotion classification.
Ensures standardized face crops with eyes aligned horizontally.
"""
import cv2
import numpy as np
from typing import Tuple, Optional

# Standard 5-point landmark template for 112x112 face crop (InsightFace / OpenCV FER standard)
STANDARD_FACIAL_5KPS_112 = np.array([
    [38.2946, 51.6963],  # Left Eye
    [73.5318, 51.5014],  # Right Eye
    [56.0252, 71.7366],  # Nose Tip
    [41.5493, 92.3655],  # Left Mouth Corner
    [70.7299, 92.2041]   # Right Mouth Corner
], dtype=np.float32)

class FaceAligner:
    """
    Performs 5-point geometric face alignment and standard preprocessing.
    """
    def __init__(self, target_size: Tuple[int, int] = (112, 112)):
        self.target_size = target_size

    def align_face(
        self,
        frame: np.ndarray,
        kps: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Aligns a human face geometrically using 5 facial keypoints.
        
        Args:
            frame (np.ndarray): Full BGR image frame.
            kps (np.ndarray): Shape (5, 2) facial keypoints.
            bbox (Tuple[int, int, int, int], optional): Fallback bounding box (x, y, w, h).
            
        Returns:
            np.ndarray: Aligned BGR face chip of shape (112, 112, 3).
        """
        if frame is None or frame.size == 0:
            return np.zeros((*self.target_size, 3), dtype=np.uint8)

        h_frame, w_frame = frame.shape[:2]

        # Use generous face bounding box crop to capture eyes, nose, mouth, and chin fully
        if bbox is not None:
            x, y, w, h = bbox
            pad_w = int(w * 0.20)
            pad_h_top = int(h * 0.15)
            pad_h_bottom = int(h * 0.35)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h_top)
            x2 = min(w_frame, x + w + pad_w)
            y2 = min(h_frame, y + h + pad_h_bottom)
            chip = frame[y1:y2, x1:x2]
            if chip.size > 0:
                return cv2.resize(chip, self.target_size)

        return np.zeros((*self.target_size, 3), dtype=np.uint8)

    def preprocess_aligned_face(self, aligned_chip: np.ndarray) -> np.ndarray:
        """
        Standardized face preprocessing for OpenCV MobileFaceNet FER ONNX inference:
        Scale=1.0/128.0, Mean=(127.5, 127.5, 127.5), SwapRB=True (RGB format).
        Input shape: (1, 3, 112, 112) normalized float blob.
        """
        if aligned_chip is None or aligned_chip.size == 0:
            aligned_chip = np.zeros((*self.target_size, 3), dtype=np.uint8)
            
        if aligned_chip.shape[:2] != self.target_size:
            aligned_chip = cv2.resize(aligned_chip, self.target_size)
            
        # Normalized MobileFaceNet blob: (1, 3, 112, 112)
        blob = cv2.dnn.blobFromImage(
            aligned_chip,
            1.0 / 128.0,
            self.target_size,
            (127.5, 127.5, 127.5),
            swapRB=True
        )
        return blob
