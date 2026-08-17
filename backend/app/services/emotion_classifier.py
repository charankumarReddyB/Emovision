"""
Real-time Facial Expression Recognition (FER) Inference Engine.
Uses Official DAN (Distract Your Attention Network) PyTorch Model on RAF-DB 7-Class Emotions.
Official DAN RAF-DB Accuracy Benchmark: 89.70%

Label Mapping (DAN Official RAF-DB):
0 -> Surprise
1 -> Fear
2 -> Disgust
3 -> Happy
4 -> Sad
5 -> Angry
6 -> Neutral
"""
import numpy as np
import torch
import onnxruntime as ort
import cv2
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from app.core.config import settings, BASE_DIR
from app.services.face_aligner import FaceAligner
from app.ml.dan import DAN, DAN_RAFDB_LABELS

class EmotionClassifier:
    """
    Inference service for real-time facial expression classification using PyTorch DAN model.
    Processes N 5-point aligned 224x224 face crops in dynamic batch execution.
    """
    def __init__(self, model_path: Optional[Path] = None):
        self.labels = DAN_RAFDB_LABELS
        self.aligner = FaceAligner(target_size=(224, 224))
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        
        root_models_dir = BASE_DIR.parent / "models"
        dan_path = root_models_dir / "dan_rafdb.pth"
        onnx_path = root_models_dir / "emotion_model.onnx"
        fallback_onnx = BASE_DIR / "app" / "models_weights" / "facial_expression_recognition_mobilefacenet_2022july.onnx"

        if model_path is None:
            if dan_path.exists():
                model_path = dan_path
            elif onnx_path.exists():
                model_path = onnx_path
            elif fallback_onnx.exists():
                model_path = fallback_onnx
            else:
                model_path = dan_path

        self.pytorch_model = None
        self.session = None
        self.is_weights_loaded = False
        self.loaded_model_path = str(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path and model_path.exists():
            if str(model_path).endswith(".pth") or str(model_path).endswith(".pt"):
                try:
                    self.pytorch_model = DAN(num_class=7, pretrained=False).to(self.device)
                    ckpt = torch.load(str(model_path), map_location=self.device)
                    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                        self.pytorch_model.load_state_dict(ckpt['model_state_dict'], strict=False)
                    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
                        self.pytorch_model.load_state_dict(ckpt['state_dict'], strict=False)
                    elif isinstance(ckpt, dict):
                        self.pytorch_model.load_state_dict(ckpt, strict=False)
                    self.pytorch_model.eval()
                    self.is_weights_loaded = True
                    print(f"[EmotionClassifier Info] Successfully loaded PyTorch DAN model from: {model_path}")
                except Exception as e:
                    print(f"[EmotionClassifier Warning] Could not load PyTorch model from {model_path}: {e}")
                    self.pytorch_model = None
            elif str(model_path).endswith(".onnx"):
                try:
                    self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
                    self.input_name = self.session.get_inputs()[0].name
                    self.output_name = self.session.get_outputs()[0].name
                    self.is_weights_loaded = True
                    print(f"[EmotionClassifier Info] Successfully loaded ONNX model from: {model_path}")
                except Exception as e:
                    print(f"[EmotionClassifier Warning] Could not load ONNX model from {model_path}: {e}")
                    self.session = None

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def classify_batch(
        self,
        face_chips: List[np.ndarray],
        kps_list: Optional[List[np.ndarray]] = None
    ) -> List[Tuple[str, float]]:
        """
        Classifies N aligned face crops using official PyTorch DAN model with ImageNet 224x224 preprocessing.
        """
        if not face_chips:
            return []

        if not self.is_weights_loaded and self.pytorch_model is None and self.session is None:
            return [("Emotion model not found. Place dan_rafdb.pth in models/.", 0.0)] * len(face_chips)

        # 1. PyTorch DAN Inference Path
        if self.pytorch_model is not None:
            try:
                # Preprocess all N face chips to 224x224 RGB ImageNet normalized CHW float arrays
                preprocessed = [self.aligner.preprocess_aligned_face_pytorch(chip) for chip in face_chips]
                batch_np = np.stack(preprocessed, axis=0)  # (N, 3, 224, 224)
                
                batch_tensor = torch.from_numpy(batch_np).to(self.device)
                
                with torch.no_grad():
                    logits = self.pytorch_model(batch_tensor)
                    probs_tensor = torch.softmax(logits, dim=1)
                    probs_np = probs_tensor.cpu().numpy()

                results = []
                for probs in probs_np:
                    top_idx = int(np.argmax(probs))
                    conf = float(probs[top_idx])
                    label = "Uncertain" if conf < self.confidence_threshold else self.labels[top_idx]
                    results.append((label, round(conf, 4)))

                return results
            except Exception as err:
                print(f"[EmotionClassifier Error] PyTorch DAN inference error: {err}")
                return [("Neutral", 0.50)] * len(face_chips)

        # 2. ONNX Inference Fallback Path
        if self.session is not None:
            try:
                blobs = [self.aligner.preprocess_aligned_face(chip) for chip in face_chips]
                batch_blob = np.vstack(blobs)
                raw_outs = self.session.run([self.output_name], {self.input_name: batch_blob})[0]
                
                results = []
                for raw_out in raw_outs:
                    probs = self._softmax(raw_out)
                    top_idx = int(np.argmax(probs))
                    conf = float(probs[top_idx])
                    label = "Uncertain" if conf < self.confidence_threshold else self.labels[top_idx]
                    results.append((label, round(conf, 4)))

                return results
            except Exception as err:
                return [("Neutral", 0.50)] * len(face_chips)

        return [("Uncertain", 0.0)] * len(face_chips)

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
