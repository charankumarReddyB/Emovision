"""
Real-time Facial Expression Recognition (FER) Inference Engine.
Uses OpenCV Zoo MobileFaceNet FER ONNX model for high-accuracy 7-emotion classification.
"""
import numpy as np
import onnxruntime as ort
import torch
import cv2
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from app.core.config import settings, BASE_DIR
from app.services.face_aligner import FaceAligner
from app.ml.dan import DAN

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
    Inference service for real-time facial expression classification.
    Processes N 5-point aligned face crops using OpenCV MobileFaceNet FER ONNX model.
    """
    def __init__(self, model_path: Optional[Path] = None):
        models_dir = BASE_DIR / "app" / "models_weights"
        onnx_colab = models_dir / "emotion_model.onnx"
        onnx_mobilefacenet = models_dir / "facial_expression_recognition_mobilefacenet_2022july.onnx"
        dan_path = models_dir / "dan_rafdb.pth"

        if model_path is None:
            if onnx_colab.exists():
                model_path = onnx_colab
            elif onnx_mobilefacenet.exists():
                model_path = onnx_mobilefacenet
            elif dan_path.exists():
                model_path = dan_path

        self.labels = OPENCV_FER_CLASSES
        self.aligner = FaceAligner(target_size=(112, 112))
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        
        self.pytorch_model = None
        self.session = None
        self.is_weights_loaded = False
        self.loaded_model_path = str(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path and model_path.exists():
            if str(model_path).endswith(".onnx"):
                try:
                    opts = ort.SessionOptions()
                    opts.log_severity_level = 3
                    self.session = ort.InferenceSession(str(model_path), opts, providers=['CPUExecutionProvider'])
                    self.input_name = self.session.get_inputs()[0].name
                    self.output_name = self.session.get_outputs()[0].name
                    self.is_weights_loaded = True
                    print(f"[EmotionClassifier Info] Successfully loaded ONNX FER model from: {model_path}")
                except Exception as e:
                    print(f"[EmotionClassifier Warning] Could not load ONNX model from {model_path}: {e}")
                    self.session = None
            elif str(model_path).endswith(".pth") or str(model_path).endswith(".pt"):
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
                    self.aligner = FaceAligner(target_size=(224, 224))
                    self.is_weights_loaded = True
                    print(f"[EmotionClassifier Info] Successfully loaded PyTorch model from: {model_path}")
                except Exception as e:
                    print(f"[EmotionClassifier Warning] Could not load PyTorch model: {e}")
                    self.pytorch_model = None

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def classify_batch(
        self,
        face_chips: List[np.ndarray],
        kps_list: Optional[List[np.ndarray]] = None
    ) -> List[Tuple[str, float]]:
        """
        Classifies N aligned face crops using ONNX FER / PyTorch model.
        """
        if not face_chips:
            return []

        # 1. ONNX Model Inference Path
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
            except Exception:
                # Per-face 1-by-1 fallback if fixed batch size ONNX model
                results = []
                for chip in face_chips:
                    blob = self.aligner.preprocess_aligned_face(chip)
                    try:
                        raw_out = self.session.run([self.output_name], {self.input_name: blob})[0][0]
                        probs = self._softmax(raw_out)
                        top_idx = int(np.argmax(probs))
                        conf = float(probs[top_idx])
                        label = "Uncertain" if conf < self.confidence_threshold else self.labels[top_idx]
                        results.append((label, round(conf, 4)))
                    except Exception:
                        results.append(("Neutral", 0.50))
                return results

        # 2. PyTorch DAN Inference Path
        if self.pytorch_model is not None:
            try:
                preprocessed = [self.aligner.preprocess_aligned_face_pytorch(chip) for chip in face_chips]
                batch_np = np.stack(preprocessed, axis=0)
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
