"""
Face Aligner Service for Emovision CV Core.
Uses 5 facial keypoints (left_eye, right_eye, nose, left_mouth, right_mouth) to perform
geometric 2D affine transformation warping before emotion classification.
Ensures standardized face crops with eyes aligned horizontally.
"""
import cv2
import numpy as np
from typing import Tuple, Optional

# Standard 5-point landmark template for 112x112 (InsightFace / OpenCV FER standard)
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
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        scale = float(target_size[0]) / 112.0
        self.reference_template = STANDARD_FACIAL_5KPS_112 * scale

    def align_face(
        self,
        frame: np.ndarray,
        kps: Optional[np.ndarray] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Aligns a human face geometrically using 5 facial keypoints.
        Uses 2D similarity affine transformation matrix estimation to level eyes horizontally.
        """
        if frame is None or frame.size == 0:
            return np.zeros((*self.target_size, 3), dtype=np.uint8)

        h_frame, w_frame = frame.shape[:2]

        # 1. Primary path: 5-point geometric affine transformation matrix estimate
        if kps is not None and isinstance(kps, np.ndarray) and kps.shape == (5, 2):
            try:
                src_pts = kps.astype(np.float32)
                dst_pts = self.reference_template.astype(np.float32)
                
                # Estimate 2D Similarity Affine Transform (rotation, translation, scale)
                M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                if M is not None:
                    aligned = cv2.warpAffine(frame, M, self.target_size, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
                    if aligned is not None and aligned.size > 0:
                        return aligned
            except Exception as err:
                print(f"[FaceAligner Warning] 5-point affine alignment fallback: {err}")

        # 2. Fallback path: Tight bounding box crop centered on face (no neck/chest distortion)
        if bbox is not None:
            x, y, w, h = bbox
            pad_w = int(w * 0.05)
            pad_h = int(h * 0.05)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(w_frame, x + w + pad_w)
            y2 = min(h_frame, y + h + pad_h)
            chip = frame[y1:y2, x1:x2]
            if chip.size > 0:
                return cv2.resize(chip, self.target_size)

        return np.zeros((*self.target_size, 3), dtype=np.uint8)

    def preprocess_aligned_face(self, aligned_chip: np.ndarray) -> np.ndarray:
        """
        Standardized face preprocessing for ONNX models:
        Scale=1.0/128.0, Mean=(127.5, 127.5, 127.5), SwapRB=True (RGB format).
        """
        if aligned_chip is None or aligned_chip.size == 0:
            aligned_chip = np.zeros((*self.target_size, 3), dtype=np.uint8)
            
        if aligned_chip.shape[:2] != self.target_size:
            aligned_chip = cv2.resize(aligned_chip, self.target_size)
            
        blob = cv2.dnn.blobFromImage(
            aligned_chip,
            1.0 / 128.0,
            self.target_size,
            (127.5, 127.5, 127.5),
            swapRB=True
        )
        return blob

    def preprocess_aligned_face_pytorch(self, aligned_chip: np.ndarray) -> np.ndarray:
        """
        Official DAN ImageNet RGB Preprocessing:
        Target Size: (224, 224, 3) RGB
        ImageNet Normalization: Mean [0.485, 0.456, 0.406], Std [0.229, 0.224, 0.225]
        Returns: (3, 224, 224) float32 numpy array
        """
        if aligned_chip is None or aligned_chip.size == 0:
            aligned_chip = np.zeros((224, 224, 3), dtype=np.uint8)

        # Convert BGR to RGB and resize to 224x224
        rgb_chip = cv2.cvtColor(aligned_chip, cv2.COLOR_BGR2RGB)
        if rgb_chip.shape[:2] != (224, 224):
            rgb_chip = cv2.resize(rgb_chip, (224, 224))

        # Normalize pixel range [0, 1]
        img = rgb_chip.astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        # HWC -> CHW (3, 224, 224)
        return np.transpose(img, (2, 0, 1))
