"""
Real-time Facial Expression Recognition (FER) Inference Engine.
Performs 7-emotion classification on cropped face regions.
Supports EmotionCNN, ResNetEmotionCNN, MobileNetV3Emotion, and EfficientNetB0Emotion model weights.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from app.core.config import settings
from app.ml.model import get_model, EmotionCNN, ResNetEmotionCNN, MobileNetV3Emotion, EfficientNetB0Emotion
from app.services.face_aligner import FaceAligner

class EmotionClassifier:
    """
    Inference service for real-time facial expression classification.
    Processes cropped face chips and predicts emotion label + confidence score.
    """
    def __init__(self, model_path: Optional[Path] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = settings.EMOTION_CLASSES
        self.aligner = FaceAligner()
        
        if model_path is None:
            model_path = settings.MODELS_DIR / "emotion_model.pth"
            
        self.model = get_model(num_classes=len(self.labels), pretrained_path=str(model_path) if model_path.exists() else None).to(self.device)
        self.is_weights_loaded = model_path.exists()
        self.model.eval()

    def classify_batch(self, face_chips: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Classifies a batch of N aligned face chips in a single PyTorch batch pass.
        
        Args:
            face_chips (List[np.ndarray]): List of N BGR face chips (48, 48, 3).
            
        Returns:
            List[Tuple[str, float]]: List of (Emotion Label, Confidence Score) per face.
        """
        if not face_chips:
            return []

        # Preprocess list of N aligned face chips into PyTorch batch tensor (N, 1, 48, 48)
        batch_tensor = self.aligner.preprocess_batch(face_chips).to(self.device)
        
        with torch.no_grad():
            logits = self.model(batch_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            
        results = []
        for p in probs:
            top_idx = int(np.argmax(p))
            conf = float(p[top_idx])
            label = self.labels[top_idx]
            results.append((label, round(conf, 4)))
            
        return results

    def predict_face(self, face_chip: np.ndarray) -> Tuple[str, float]:
        """Classifies a single cropped BGR face chip."""
        if face_chip is None or face_chip.size == 0:
            return "Neutral", 0.50
        res = self.classify_batch([face_chip])
        return res[0] if res else ("Neutral", 0.50)

    def classify_face(self, face_chip: np.ndarray) -> Tuple[str, float]:
        """Alias for predict_face."""
        return self.predict_face(face_chip)

    def classify_tracked_faces(
        self,
        detections: List[Dict[str, Any]],
        frame_idx: int = 0
    ) -> List[Dict[str, Any]]:
        """Classifies facial expressions for N detected faces."""
        chips = [det.get("aligned_chip", det.get("face_chip")) for det in detections]
        preds = self.classify_batch(chips)
        for det, (label, conf) in zip(detections, preds):
            det["emotion"] = label
            det["emotion_confidence"] = conf
        return detections
