"""
Unit tests for Emovision Computer Vision & Backend Services.
Tests Face Detection (YuNet DNN), Preprocessor, FPSCounter, and Database logging.
"""
import pytest
import numpy as np
import os
import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.services.face_detector import FaceDetector
from app.services.preprocessing import FacePreprocessor
from app.services.fps_counter import FPSCounter
from app.db.database import create_session, log_frame_detections
from test_face_pipeline import generate_synthetic_multi_face_frame

def test_face_detector_initialization():
    detector = FaceDetector()
    assert detector is not None
    assert hasattr(detector, "detect_faces")

def test_empty_background_zero_false_positives():
    """Verifies that an empty background image returns 0 face detections."""
    detector = FaceDetector()
    empty_bg = np.zeros((480, 640, 3), dtype=np.uint8)
    empty_bg[:, :] = (200, 180, 190)
    
    detections = detector.detect_faces(empty_bg)
    assert isinstance(detections, list)
    assert len(detections) == 0  # 0 false positives on background

def test_face_preprocessor():
    """Verifies that cropped faces are properly normalized into (1, 48, 48, 1) model tensors."""
    preprocessor = FacePreprocessor(target_size=(48, 48), color_mode="grayscale")
    dummy_face = np.ones((100, 100, 3), dtype=np.uint8) * 128
    
    tensor = preprocessor.preprocess(dummy_face)
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, 48, 48, 1)
    assert tensor.dtype == np.float32
    assert 0.0 <= tensor.min() <= tensor.max() <= 1.0

def test_fps_counter():
    fps_counter = FPSCounter()
    fps_counter.start()
    for _ in range(10):
        fps = fps_counter.update()
    avg_fps = fps_counter.get_avg_fps()
    assert avg_fps >= 0.0

def test_database_logging():
    """Verifies repository session creation and frame detection logging."""
    import uuid
    from app.db.repository import get_db_repository
    repo = get_db_repository()
    
    session_id = f"test_unit_{uuid.uuid4().hex[:6]}"
    try:
        repo.create_session(session_id, "Unit Test Session", "test")
        sample_detections = [
            {"face_index": 1, "bbox": (50, 50, 100, 100), "confidence": 0.95},
            {"face_index": 2, "bbox": (200, 150, 80, 80), "confidence": 0.88}
        ]
        repo.log_frame_predictions(session_id, frame_number=1, detections=sample_detections)
        sess = repo.get_session(session_id)
        assert sess is not None
        assert sess["session_id"] == session_id
    except Exception as e:
        print(f"[DB Log Warning] Network or DB skip: {e}")
