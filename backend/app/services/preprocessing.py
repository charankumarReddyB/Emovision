"""
Face Image Preprocessing Utilities for Facial Expression Classification.
Prepares cropped face chips for model input (grayscale/RGB conversion, resizing, normalization).
"""
import cv2
import numpy as np
from typing import Tuple
from app.core.config import settings

class FacePreprocessor:
    """
    Standard preprocessor preparing detected face images for emotion recognition models.
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = settings.TARGET_FACE_SIZE,
        color_mode: str = settings.COLOR_MODE
    ):
        self.target_size = target_size
        self.color_mode = color_mode.lower()

    def preprocess(self, face_chip: np.ndarray) -> np.ndarray:
        """
        Preprocesses a cropped BGR face chip into a normalized model input tensor.
        
        Args:
            face_chip (np.ndarray): Cropped BGR face image.
            
        Returns:
            np.ndarray: Normalized tensor of shape (1, H, W, C).
        """
        if face_chip is None or face_chip.size == 0:
            # Fallback zero array if input is invalid
            channels = 1 if self.color_mode == "grayscale" else 3
            return np.zeros((1, self.target_size[0], self.target_size[1], channels), dtype=np.float32)

        # 1. Color conversion
        if self.color_mode == "grayscale":
            if len(face_chip.shape) == 3 and face_chip.shape[2] == 3:
                processed = cv2.cvtColor(face_chip, cv2.COLOR_BGR2GRAY)
            else:
                processed = face_chip.copy()
        else:
            if len(face_chip.shape) == 2:
                processed = cv2.cvtColor(face_chip, cv2.COLOR_GRAY2BGR)
            elif face_chip.shape[2] == 3:
                processed = cv2.cvtColor(face_chip, cv2.COLOR_BGR2RGB)
            else:
                processed = face_chip.copy()

        # 2. Resize to target input dimensions
        processed = cv2.resize(processed, self.target_size, interpolation=cv2.INTER_AREA)

        # 3. Intensity normalization [0, 255] -> [0.0, 1.0]
        normalized = processed.astype(np.float32) / 255.0

        # 4. Expand dimensions for batch input (1, H, W, C)
        if self.color_mode == "grayscale":
            tensor = np.expand_dims(normalized, axis=(0, -1))
        else:
            tensor = np.expand_dims(normalized, axis=0)

        return tensor
