"""
Real-time Facial Expression Recognition (FER) Inference Engine.
Loads trained EmotionCNN model weights and performs 7-emotion classification on cropped face regions.
Integrates per-Person ID prediction caching for real-time high-FPS video streaming.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from app.core.config import settings
from app.ml.model import EmotionCNN
from app.ml.dataset import LABEL_MAP
from app.services.preprocessing import FacePreprocessor

class EmotionClassifier:
    """
    Inference service for real-time facial expression classification.
    Processes cropped face chips and predicts emotion label + confidence.
    """
    def __init__(self, model_path: Optional[Path] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = settings.EMOTION_CLASSES
        self.preprocessor = FacePreprocessor(target_size=(48, 48), color_mode="grayscale")
        
        # Determine model weights path
        if model_path is None:
            model_path = settings.MODELS_DIR / "emotion_model.pth"
            
        self.model = EmotionCNN(num_classes=len(self.labels)).to(self.device)
        self.is_weights_loaded = False
        
        if model_path.exists():
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.is_weights_loaded = True
            except Exception as e:
                print(f"[EmotionClassifier Warning] Could not load weights from {model_path}: {e}")
                
        self.model.eval()
        
        # Per-person ID prediction cache for real-time FPS optimization
        self.prediction_cache: Dict[int, Dict[str, Any]] = {}
        self.cache_ttl_frames = 3  # Re-evaluate every 3 frames per tracked person ID

    def predict_face(self, face_chip: np.ndarray) -> Tuple[str, float]:
        """
        Classifies a single cropped BGR face chip into 1 of 7 emotion categories.
        
        Args:
            face_chip (np.ndarray): Cropped BGR face image.
            
        Returns:
            Tuple[str, float]: (Emotion Label, Confidence Score [0.0 - 1.0])
        """
        if face_chip is None or face_chip.size == 0:
            return "Neutral", 0.50

        # Preprocess face chip to tensor (1, 48, 48, 1)
        tensor = self.preprocessor.preprocess(face_chip)
        
        # Transpose from (B, H, W, C) to PyTorch layout (B, C, H, W)
        if len(tensor.shape) == 4 and tensor.shape[-1] in (1, 3):
            tensor = np.transpose(tensor, (0, 3, 1, 2))

        torch_tensor = torch.from_numpy(tensor).to(self.device)
        
        # Standardize range [-1.0, 1.0] matching training transform
        torch_tensor = (torch_tensor - 0.5) / 0.5

        with torch.no_grad():
            logits = self.model(torch_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        emotion_label = self.labels[top_idx]

        return emotion_label, round(confidence, 4)

    def classify_tracked_faces(
        self,
        tracked_detections: List[Dict[str, Any]],
        frame_idx: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Classifies facial expressions for N tracked faces dynamically.
        Uses person_id prediction caching to maintain high real-time FPS.
        
        Args:
            tracked_detections (List[Dict[str, Any]]): Face detections with assigned person_id.
            frame_idx (int): Current frame index.
            
        Returns:
            List[Dict[str, Any]]: Detections enriched with 'emotion' and 'emotion_confidence'.
        """
        for det in tracked_detections:
            pid = det.get("person_id", -1)
            face_chip = det.get("face_chip")

            # Check cache for person_id to avoid redundant inference on every frame
            if pid > 0 and pid in self.prediction_cache:
                cached_data = self.prediction_cache[pid]
                if (frame_idx - cached_data["last_frame"]) < self.cache_ttl_frames:
                    det["emotion"] = cached_data["emotion"]
                    det["emotion_confidence"] = cached_data["confidence"]
                    continue

            # Run inference
            label, conf = self.predict_face(face_chip)
            det["emotion"] = label
            det["emotion_confidence"] = conf

            # Cache prediction
            if pid > 0:
                self.prediction_cache[pid] = {
                    "emotion": label,
                    "confidence": conf,
                    "last_frame": frame_idx
                }

        return tracked_detections
