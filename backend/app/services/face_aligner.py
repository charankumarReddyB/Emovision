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

# Standard 5-point landmark template for 48x48 face crop
STANDARD_FACIAL_5KPS_48 = np.array([
    [15.0, 17.0],  # Left Eye
    [33.0, 17.0],  # Right Eye
    [24.0, 27.0],  # Nose Tip
    [17.0, 37.0],  # Left Mouth Corner
    [31.0, 37.0]   # Right Mouth Corner
], dtype=np.float32)

class FaceAligner:
    """
    Performs 5-point geometric face alignment and standard preprocessing.
    """
    def __init__(self, target_size: Tuple[int, int] = (48, 48)):
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
            np.ndarray: Aligned BGR face chip of shape (48, 48, 3).
        """
        if frame is None or frame.size == 0:
            return np.zeros((*self.target_size, 3), dtype=np.uint8)

        if kps is not None and len(kps) == 5:
            src_pts = kps.astype(np.float32)
            dst_pts = STANDARD_FACIAL_5KPS_48.copy()
            
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
        Standardized face preprocessing for PyTorch inference:
        1. BGR -> Grayscale (48, 48)
        2. Normalization: (img / 255.0 - 0.5) / 0.5  (range -1.0 to +1.0)
        3. Shape: (1, 1, 48, 48) numpy array
        
        Args:
            aligned_chip (np.ndarray): BGR face chip (48, 48, 3).
            
        Returns:
            np.ndarray: Normalized 4D array shape (1, 1, 48, 48).
        """
        if aligned_chip is None or aligned_chip.size == 0:
            aligned_chip = np.zeros((*self.target_size, 3), dtype=np.uint8)
            
        if len(aligned_chip.shape) == 3 and aligned_chip.shape[2] == 3:
            gray = cv2.cvtColor(aligned_chip, cv2.COLOR_BGR2GRAY)
        else:
            gray = aligned_chip.copy()
            
        if gray.shape[:2] != self.target_size:
            gray = cv2.resize(gray, self.target_size)
            
        normalized = (gray.astype(np.float32) / 255.0 - 0.5) / 0.5
        return normalized.reshape(1, 1, 48, 48)

    def preprocess_batch(self, aligned_chips: list) -> torch.Tensor:
        """
        Preprocesses a list of N aligned face chips into a batch PyTorch tensor (N, 1, 48, 48).
        """
        if not aligned_chips:
            return torch.empty((0, 1, 48, 48), dtype=torch.float32)
            
        processed_list = [self.preprocess_aligned_face(chip) for chip in aligned_chips]
        batch_np = np.vstack(processed_list)
        return torch.from_numpy(batch_np).float()
