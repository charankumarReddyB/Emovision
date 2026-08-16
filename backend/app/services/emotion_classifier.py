"""
Real-time Facial Expression Recognition (FER) Inference Engine.
Performs 7-emotion classification on cropped face regions.
Supports both EmotionCNN and ResNetEmotionCNN model weights.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from app.core.config import settings
from app.ml.model import get_model, EmotionCNN, ResNetEmotionCNN
from app.ml.dataset import LABEL_MAP
from app.services.preprocessing import FacePreprocessor

class EmotionClassifier:
    """
    Inference service for real-time facial expression classification.
    Processes cropped face chips and predicts emotion label + confidence score.
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
                state_dict = torch.load(model_path, map_location=self.device)
                if any(k.startswith('in_conv') for k in state_dict.keys()):
                    self.model = ResNetEmotionCNN(num_classes=len(self.labels)).to(self.device)
                self.model.load_state_dict(state_dict)
                self.is_weights_loaded = True
            except Exception as e:
                print(f"[EmotionClassifier Warning] Could not load weights from {model_path}: {e}")
                
        self.model.eval()

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

    def classify_face(self, face_chip: np.ndarray) -> Tuple[str, float]:
        """Alias for predict_face."""
        return self.predict_face(face_chip)

    def classify_tracked_faces(
        self,
        detections: List[Dict[str, Any]],
        frame_idx: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Classifies facial expressions for N detected faces.
        """
        for det in detections:
            face_chip = det.get("face_chip")
            label, conf = self.predict_face(face_chip)
            det["emotion"] = label
            det["emotion_confidence"] = conf

        return detections
