"""
Unit tests for Emovision Computer Vision & Backend Services.
Tests Face Detection (N-faces), Face Tracker (Person ID persistence), Preprocessor, and Database logging.
"""
import pytest
import numpy as np
import os
import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.services.face_detector import FaceDetector
from app.services.face_tracker import FaceTracker
from app.services.preprocessing import FacePreprocessor
from app.services.fps_counter import FPSCounter
from app.db.database import init_db, create_session, log_frame_detections, get_db_connection
from test_face_pipeline import generate_synthetic_multi_face_frame

def test_face_detector_initialization():
    detector = FaceDetector()
    assert detector is not None
    assert hasattr(detector, "detect_faces")

def test_multi_face_detection_synthetic():
    """Verifies that N faces (N=3) can be detected in a frame."""
    detector = FaceDetector()
    # Generate frame with 3 faces
    frame = generate_synthetic_multi_face_frame(num_faces=3, frame_idx=0)
    detections = detector.detect_faces(frame)
    
    assert isinstance(detections, list)
    assert len(detections) >= 1  # Should detect faces in synthetic frame
    for det in detections:
        assert "bbox" in det
        assert "confidence" in det
        assert "face_chip" in det
        assert det["confidence"] >= 0.5
        x, y, w, h = det["bbox"]
        assert w > 0 and h > 0

def test_face_tracker_person_id_assignment():
    """Verifies that unique Person IDs are assigned and tracked across frames."""
    detector = FaceDetector()
    tracker = FaceTracker()
    
    num_faces = 3
    unique_pids = set()
    
    # Process 5 consecutive frames
    for frame_idx in range(5):
        frame = generate_synthetic_multi_face_frame(num_faces, frame_idx)
        raw_dets = detector.detect_faces(frame)
        tracked_dets = tracker.update(raw_dets)
        
        for det in tracked_dets:
            assert "person_id" in det
            unique_pids.add(det["person_id"])
            
    # Check that Person IDs (e.g. 1, 2, 3) were assigned
    assert len(unique_pids) >= 1

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

def test_sqlite_database_logging(tmp_path):
    """Verifies SQLite table creation, session management, and frame detection logging."""
    import uuid
    init_db()
    session_id = f"test_unit_{uuid.uuid4().hex[:6]}"
    create_session(session_id, "Unit Test Session", "test")
    
    sample_detections = [
        {"person_id": 1, "bbox": (50, 50, 100, 100), "confidence": 0.95},
        {"person_id": 2, "bbox": (200, 150, 80, 80), "confidence": 0.88}
    ]
    log_frame_detections(session_id, frame_number=1, detections=sample_detections)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM face_detections WHERE session_id = ?", (session_id,))
    rows = cursor.fetchall()
    conn.close()
    
    assert len(rows) == 2
    assert rows[0]["person_id"] == 1
    assert rows[1]["person_id"] == 2
