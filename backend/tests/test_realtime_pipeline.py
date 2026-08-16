"""
Unit & Performance Tests for Real-Time Multi-Face Facial Expression Recognition Pipeline.
Tests:
1. Session tracker statistics and logging
2. Real-time pipeline processing with 1, 2, and N faces
3. FPS and execution latency measurements under 50ms (> 20 FPS)
"""
import pytest
import numpy as np
import time
import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.services.session_tracker import SessionTracker
from app.services.realtime_pipeline import RealtimePipeline
from test_face_pipeline import generate_synthetic_multi_face_frame

def test_session_tracker_stats():
    """Tests session tracker live statistics."""
    tracker = SessionTracker("test_tracker_sess", "Unit Test Session", "test")
    
    sample_dets = [
        {"face_index": 1, "bbox": (50, 50, 60, 60), "confidence": 0.9, "emotion": "Happy", "emotion_confidence": 0.92},
        {"face_index": 2, "bbox": (200, 100, 70, 70), "confidence": 0.85, "emotion": "Happy", "emotion_confidence": 0.88},
        {"face_index": 3, "bbox": (350, 200, 50, 50), "confidence": 0.8, "emotion": "Neutral", "emotion_confidence": 0.75}
    ]
    
    stats = tracker.process_frame_detections(frame_number=1, detections=sample_dets)
    
    assert stats["total_people"] == 3
    assert stats["dominant_expression"] == "Happy"
    assert stats["expression_counts"]["Happy"] == 2
    assert stats["expression_counts"]["Neutral"] == 1
    assert stats["average_confidence"] > 80.0

def test_realtime_pipeline_multi_face():
    """Tests master pipeline processing N faces dynamically."""
    pipeline = RealtimePipeline(session_id="test_pipe_sess")
    frame = generate_synthetic_multi_face_frame(num_faces=3, frame_idx=0)
    hud_frame, stats = pipeline.process_frame(frame, frame_idx=0)
    
    assert isinstance(hud_frame, np.ndarray)
    assert hud_frame.shape == frame.shape
    assert "fps" in stats
    assert "dominant_expression" in stats

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
