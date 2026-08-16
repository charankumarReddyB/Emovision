"""
Face Aligner Service for Emovision CV Core.
Uses 5 facial keypoints (left_eye, right_eye, nose, left_mouth, right_mouth) to perform
geometric 2D affine transformation warping before emotion classification.
Ensures standardized face crops with eyes aligned horizontally.
"""
import cv2
import numpy as np
import torch
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

        if kps is not None and len(kps) == 5:
            src_pts = kps.astype(np.float32)
            dst_pts = STANDARD_FACIAL_5KPS_112.copy()
            
            # Estimate partial affine matrix (rotation, uniform scale, translation)
            M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
            
            if M is not None:
                aligned_chip = cv2.warpAffine(
                    frame,
                    M,
                    self.target_size,
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REFLECT
                )
                return aligned_chip

        # Fallback to standard bounding box crop if keypoints are unavailable
        if bbox is not None:
            x, y, w, h = bbox
            h_frame, w_frame = frame.shape[:2]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w_frame, x + w)
            y2 = min(h_frame, y + h)
            chip = frame[y1:y2, x1:x2]
            if chip.size > 0:
                return cv2.resize(chip, self.target_size)
                
        return np.zeros((*self.target_size, 3), dtype=np.uint8)

    def preprocess_aligned_face(self, aligned_chip: np.ndarray) -> np.ndarray:
        """
        Standardized face preprocessing for OpenCV MobileFaceNet FER ONNX inference:
        Input shape: (1, 3, 112, 112) BGR float array.
        """
        if aligned_chip is None or aligned_chip.size == 0:
            aligned_chip = np.zeros((*self.target_size, 3), dtype=np.uint8)
            
        if aligned_chip.shape[:2] != self.target_size:
            aligned_chip = cv2.resize(aligned_chip, self.target_size)
            
        # Blob from BGR image: (1, 3, 112, 112)
        blob = cv2.dnn.blobFromImage(aligned_chip, 1.0, self.target_size, (0.0, 0.0, 0.0), swapRB=False)
        return blob
