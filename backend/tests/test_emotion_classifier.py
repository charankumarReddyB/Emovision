"""
Unit Tests for Facial Expression Recognition (FER) Module.
Tests:
1. Single face emotion prediction
2. Multiple faces in one frame
3. Different expression classes (7 target emotions)
4. Invalid/empty image input
5. Model loading & weights initialization
6. Confidence score output bounds (0.0 to 1.0)
7. Real-time inference speed
"""
import pytest
import numpy as np
import torch
import time
import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.ml.model import EmotionCNN
from app.services.emotion_classifier import EmotionClassifier
from test_face_pipeline import generate_synthetic_multi_face_frame

def test_model_loading_and_architecture():
    """Test 5: Model loading & architecture structure."""
    classifier = EmotionClassifier()
    assert classifier.model is not None
    assert isinstance(classifier.model, torch.nn.Module)
    assert classifier.labels == settings.EMOTION_CLASSES
    assert len(classifier.labels) == 7

def test_single_face_emotion_prediction():
    """Test 1: Single face emotion prediction."""
    classifier = EmotionClassifier()
    dummy_face = np.ones((60, 60, 3), dtype=np.uint8) * 150
    
    emotion_label, conf = classifier.predict_face(dummy_face)
    
    assert isinstance(emotion_label, str)
    assert emotion_label in settings.EMOTION_CLASSES
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0

def test_confidence_output_bounds():
    """Test 6: Confidence score output bounds."""
    classifier = EmotionClassifier()
    for _ in range(5):
        random_face = np.random.randint(0, 256, (48, 48, 3), dtype=np.uint8)
        label, conf = classifier.predict_face(random_face)
        assert 0.0 <= conf <= 1.0
        assert label in settings.EMOTION_CLASSES

def test_multiple_faces_in_one_frame():
    """Test 2: Multiple faces in one frame (N faces)."""
    classifier = EmotionClassifier()
    
    sample_detections = [
        {"face_index": 1, "bbox": (50, 50, 60, 60), "face_chip": np.ones((60, 60, 3), dtype=np.uint8)*100},
        {"face_index": 2, "bbox": (200, 100, 70, 70), "face_chip": np.ones((70, 70, 3), dtype=np.uint8)*180},
        {"face_index": 3, "bbox": (350, 200, 50, 50), "face_chip": np.ones((50, 50, 3), dtype=np.uint8)*220}
    ]
    
    classified = classifier.classify_tracked_faces(sample_detections, frame_idx=0)
    
    assert len(classified) == 3
    for det in classified:
        assert "emotion" in det
        assert "emotion_confidence" in det
        assert det["emotion"] in settings.EMOTION_CLASSES
        assert 0.0 <= det["emotion_confidence"] <= 1.0

def test_different_expression_classes():
    """Test 3: Different expression classes (7 FER emotions)."""
    labels = settings.EMOTION_CLASSES
    expected_classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    
    for cls_name in expected_classes:
        assert cls_name in labels

def test_invalid_empty_image_input():
    """Test 4: Invalid/empty image input (graceful handling)."""
    classifier = EmotionClassifier()
    
    # Test None input
    label, conf = classifier.predict_face(None)
    assert label == "Neutral"
    assert conf == 0.50
    
    # Test 0-byte array
    empty_face = np.array([], dtype=np.uint8)
    label_empty, conf_empty = classifier.predict_face(empty_face)
    assert label_empty == "Neutral"
    assert conf_empty == 0.50

def test_realtime_inference_speed():
    """Test 7: Real-time inference speed."""
    classifier = EmotionClassifier()
    dummy_chip = np.ones((60, 60, 3), dtype=np.uint8)*120
    
    t0 = time.perf_counter()
    for _ in range(10):
        label, conf = classifier.predict_face(dummy_chip)
    elapsed_ms = (time.perf_counter() - t0) * 100.0
    assert elapsed_ms < 1000.0
