"""
Real-time Facial Expression Recognition (FER) ONNX Inference Engine.
Uses OpenCV Zoo MobileFaceNet FER ONNX model for high-accuracy 7-emotion classification.
Displays 'Uncertain' if prediction confidence is below configurable threshold.
"""
import numpy as np
import onnxruntime as ort
import cv2
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from app.core.config import settings
from app.services.face_aligner import FaceAligner

# Exact OpenCV MobileFaceNet FER ONNX Class Labels:
OPENCV_FER_CLASSES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

class EmotionClassifier:
    """
    ONNX Runtime Inference service for real-time facial expression classification.
    Processes N 5-point aligned face chips using OpenCV MobileFaceNet FER ONNX model.
    """
    def __init__(self, model_path: Optional[Path] = None):
        self.labels = OPENCV_FER_CLASSES
        self.aligner = FaceAligner(target_size=(112, 112))
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        
        if model_path is None:
            model_path = settings.MODELS_DIR / "facial_expression_recognition_mobilefacenet_2022july.onnx"
            if not model_path.exists():
                fallback = settings.MODELS_DIR / "emotion_model.onnx"
                if fallback.exists():
                    model_path = fallback
            
        self.session = None
        self.is_weights_loaded = False
        
        if model_path.exists():
            try:
                self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
                self.is_weights_loaded = True
            except Exception as e:
                print(f"[EmotionClassifier Warning] Could not load ONNX model from {model_path}: {e}")

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def classify_batch(self, face_chips: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Classifies N aligned face chips in ONNX Runtime.
        
        Args:
            face_chips (List[np.ndarray]): List of N BGR face chips (112, 112, 3).
            
        Returns:
            List[Tuple[str, float]]: List of (Emotion Label, Confidence Score) per face.
                Returns ('Uncertain', confidence) if confidence < CONFIDENCE_THRESHOLD.
        """
        if not face_chips or self.session is None:
            return []

        # Preprocess list of N aligned face chips into individual (1, 3, 112, 112) blobs
        blobs = [self.aligner.preprocess_aligned_face(chip) for chip in face_chips]
        
        raw_outputs = []
        for blob in blobs:
            out = self.session.run([self.output_name], {self.input_name: blob})[0]
            raw_outputs.append(out[0])
            
        outputs = np.array(raw_outputs)
        probs = self._softmax(outputs)
        
        results = []
        for p in probs:
            top_idx = int(np.argmax(p))
            conf = float(p[top_idx])
            
            # Enforce configurable confidence thresholding
            if conf < self.confidence_threshold:
                label = "Uncertain"
            else:
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
