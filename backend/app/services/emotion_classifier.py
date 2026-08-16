"""
Real-time Facial Expression Recognition (FER) ONNX Inference Engine.
Performs 7-emotion classification on cropped face regions using ONNX Runtime for high-speed batch inference.
Displays 'Uncertain' if prediction confidence is below configurable threshold.
"""
import numpy as np
import onnxruntime as ort
import cv2
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from app.core.config import settings
from app.services.face_aligner import FaceAligner

class EmotionClassifier:
    """
    ONNX Runtime Inference service for real-time facial expression classification.
    Processes N 5-point aligned face chips in a single ONNX batch pass [N, 1, 48, 48].
    """
    def __init__(self, model_path: Optional[Path] = None):
        self.labels = settings.EMOTION_CLASSES
        self.aligner = FaceAligner()
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        
        if model_path is None:
            model_path = settings.MODELS_DIR / "emotion_model.onnx"
            
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
        Classifies N aligned face chips in a single ONNX Runtime batch pass.
        
        Args:
            face_chips (List[np.ndarray]): List of N BGR face chips (48, 48, 3).
            
        Returns:
            List[Tuple[str, float]]: List of (Emotion Label, Confidence Score) per face.
                Returns ('Uncertain', confidence) if confidence < CONFIDENCE_THRESHOLD.
        """
        if not face_chips or self.session is None:
            return []

        # Preprocess list of N aligned face chips into batch numpy array (N, 1, 48, 48)
        processed_list = [self.aligner.preprocess_aligned_face(chip) for chip in face_chips]
        batch_np = np.vstack(processed_list).astype(np.float32)
        
        # Execute ONNX Runtime single-pass batch inference
        outputs = self.session.run([self.output_name], {self.input_name: batch_np})[0]
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
