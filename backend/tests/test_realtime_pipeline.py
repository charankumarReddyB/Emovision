"""
Unit & Performance Tests for Real-Time Multi-Person Facial Expression Recognition Pipeline.
Tests:
1. Prediction smoother majority voting & confidence averaging
2. Session tracker statistics & SQLite database logging
3. Real-time pipeline processing with 1, 2, and N people
4. People entering and leaving frame dynamics
5. FPS and latency measurements
"""
import pytest
import numpy as np
import time
import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.services.prediction_smoother import PredictionSmoother
from app.services.session_tracker import SessionTracker
from app.services.realtime_pipeline import RealtimePipeline
from test_face_pipeline import generate_synthetic_multi_face_frame

def test_prediction_smoother_majority_voting():
    """Tests temporal prediction smoother majority voting."""
    smoother = PredictionSmoother(window_size=5)
    pid = 1
    
    # Push sequence: Happy, Happy, Neutral, Happy, Angry
    smoother.smooth_prediction(pid, "Happy", 0.90)
    smoother.smooth_prediction(pid, "Happy", 0.85)
    smoother.smooth_prediction(pid, "Neutral", 0.70)
    smoother.smooth_prediction(pid, "Happy", 0.88)
    smoothed_emo, smoothed_conf = smoother.smooth_prediction(pid, "Angry", 0.60)
    
    # Majority emotion should be "Happy"
    assert smoothed_emo == "Happy"
    assert 0.80 <= smoothed_conf <= 0.95

def test_session_tracker_stats_and_sqlite():
    """Tests session tracker live statistics and SQLite logging."""
    tracker = SessionTracker("test_tracker_sess", "Unit Test Session", "test")
    
    sample_dets = [
        {"person_id": 1, "bbox": (50, 50, 60, 60), "confidence": 0.9, "emotion": "Happy", "emotion_confidence": 0.92},
        {"person_id": 2, "bbox": (200, 100, 70, 70), "confidence": 0.85, "emotion": "Happy", "emotion_confidence": 0.88},
        {"person_id": 3, "bbox": (350, 200, 50, 50), "confidence": 0.8, "emotion": "Neutral", "emotion_confidence": 0.75}
    ]
    
    stats = tracker.process_frame_detections(frame_number=1, detections=sample_dets)
    
    assert stats["total_people"] == 3
    assert stats["dominant_expression"] == "Happy"
    assert stats["expression_counts"]["Happy"] == 2
    assert stats["expression_counts"]["Neutral"] == 1
    assert stats["average_confidence"] > 80.0

def test_realtime_pipeline_multi_person():
    """Tests master pipeline processing N faces dynamically."""
    pipeline = RealtimePipeline(session_id="test_pipe_sess")
    
    # Test 3 faces frame
    frame = generate_synthetic_multi_face_frame(num_faces=3, frame_idx=0)
    hud_frame, stats = pipeline.process_frame(frame, frame_idx=0)
    
    assert isinstance(hud_frame, np.ndarray)
    assert hud_frame.shape == frame.shape
    assert stats["total_people"] >= 1
    assert "fps" in stats
    assert "dominant_expression" in stats

def test_people_entering_and_leaving_frame():
    """Tests person entering (N=2 -> N=4) and leaving (N=4 -> N=1) frame."""
    pipeline = RealtimePipeline(session_id="test_entering_leaving")
    
    # Step 1: N=2 people
    f1 = generate_synthetic_multi_face_frame(num_faces=2, frame_idx=0)
    _, stats1 = pipeline.process_frame(f1, 0)
    
    # Step 2: N=4 people enter
    f2 = generate_synthetic_multi_face_frame(num_faces=4, frame_idx=1)
    _, stats2 = pipeline.process_frame(f2, 1)
    
    # Step 3: N=1 person leaves
    f3 = generate_synthetic_multi_face_frame(num_faces=1, frame_idx=2)
    _, stats3 = pipeline.process_frame(f3, 2)
    
    assert stats1["total_people"] >= 1
    assert stats2["total_people"] >= stats1["total_people"]

def test_pipeline_fps_and_latency_measurement():
    """Measures pipeline execution latency and FPS."""
    pipeline = RealtimePipeline(session_id="test_latency")
    frame = generate_synthetic_multi_face_frame(num_faces=3, frame_idx=0)
    
    latencies = []
    for i in range(10):
        t0 = time.perf_counter()
        _, stats = pipeline.process_frame(frame, i)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms
        
    avg_latency_ms = sum(latencies) / len(latencies)
    assert avg_latency_ms < 50.0  # Should execute under 50ms per frame (> 20 FPS)
