"""
FastAPI REST API Endpoints Unit Test Suite.
Tests:
1. GET /api/health
2. GET /api/model
3. POST /api/session/start and POST /api/session/{session_id}/end
4. GET /api/session/{session_id}/current
5. GET /api/session/{session_id}/analytics
6. GET /api/session/{session_id}/person/{person_id}
7. GET /api/sessions (with pagination)
8. GET /api/sessions/{session_id}
9. Error handling for invalid session ID and invalid person ID
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from main import app
from app.services.session_tracker import SessionTracker

client = TestClient(app)

def test_health_endpoint():
    """Test 1: GET /api/health."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["cv_model_status"] == "ready"
    assert data["tracking_status"] == "active"
    assert "timestamp" in data

def test_model_info_endpoint():
    """Test 2: GET /api/model."""
    response = client.get("/api/model")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert len(data["emotion_classes"]) == 7
    assert data["input_face_size"] == [48, 48]
    assert "Happy" in data["emotion_classes"]

def test_session_start_and_end_lifecycle():
    """Test 3: POST /api/session/start and POST /api/session/{session_id}/end."""
    # Start Session
    start_resp = client.post("/api/session/start", json={"session_name": "API Test Session", "source_type": "api_test"})
    assert start_resp.status_code == 201
    start_data = start_resp.json()
    session_id = start_data["session_id"]
    assert session_id.startswith("sess_")
    assert start_data["status"] == "active"

    # End Session
    end_resp = client.post(f"/api/session/{session_id}/end")
    assert end_resp.status_code == 200
    end_data = end_resp.json()
    assert end_data["session_id"] == session_id
    assert "duration_seconds" in end_data
    assert "dominant_expression" in end_data

def test_current_detection_endpoint():
    """Test 4: GET /api/session/{session_id}/current."""
    # Create session and log dummy detections
    start_resp = client.post("/api/session/start", json={"session_name": "Current Detection Test"})
    session_id = start_resp.json()["session_id"]
    
    tracker = SessionTracker(session_id)
    sample_dets = [
        {"face_index": 1, "person_id": 1, "bbox": (120, 80, 150, 160), "confidence": 0.95, "emotion": "Happy", "emotion_confidence": 0.92},
        {"face_index": 2, "person_id": 2, "bbox": (300, 100, 140, 140), "confidence": 0.88, "emotion": "Neutral", "emotion_confidence": 0.85}
    ]
    tracker.process_frame_detections(frame_number=1, detections=sample_dets)
    
    curr_resp = client.get(f"/api/session/{session_id}/current")
    assert curr_resp.status_code == 200
    curr_data = curr_resp.json()
    assert curr_data["people_detected"] == 2
    assert curr_data["dominant_expression"] == "Happy"
    assert len(curr_data["people"]) == 2
    assert "person_id" in curr_data["people"][0] or "face_index" in curr_data["people"][0]
    assert curr_data["people"][0]["expression"] == "Happy"
    assert curr_data["people"][0]["bounding_box"]["x"] == 120

def test_session_analytics_endpoint():
    """Test 5: GET /api/session/{session_id}/analytics."""
    start_resp = client.post("/api/session/start", json={"session_name": "Analytics Test"})
    session_id = start_resp.json()["session_id"]
    
    tracker = SessionTracker(session_id)
    sample_dets = [
        {"face_index": 1, "person_id": 1, "bbox": (100, 100, 50, 50), "emotion": "Happy", "emotion_confidence": 0.9},
        {"face_index": 2, "person_id": 2, "bbox": (200, 200, 50, 50), "emotion": "Surprise", "emotion_confidence": 0.85}
    ]
    tracker.process_frame_detections(frame_number=1, detections=sample_dets)
    
    analytics_resp = client.get(f"/api/session/{session_id}/analytics")
    assert analytics_resp.status_code == 200
    data = analytics_resp.json()
    assert data["total_people_detected"] == 2
    assert data["total_predictions"] == 2
    assert "Happy" in data["expression_distribution"]

def test_person_analytics_endpoint():
    """Test 6: GET /api/session/{session_id}/person/{person_id}."""
    start_resp = client.post("/api/session/start", json={"session_name": "Person Analytics Test"})
    session_id = start_resp.json()["session_id"]
    
    tracker = SessionTracker(session_id)
    # Log multi-frame timeline for Person 1: Neutral -> Happy -> Surprise
    tracker.process_frame_detections(1, [{"face_index": 1, "person_id": 1, "bbox": (100, 100, 50, 50), "emotion": "Neutral", "emotion_confidence": 0.8}])
    tracker.process_frame_detections(2, [{"face_index": 1, "person_id": 1, "bbox": (100, 100, 50, 50), "emotion": "Happy", "emotion_confidence": 0.9}])
    tracker.process_frame_detections(3, [{"face_index": 1, "person_id": 1, "bbox": (100, 100, 50, 50), "emotion": "Surprise", "emotion_confidence": 0.85}])
    
    p_resp = client.get(f"/api/session/{session_id}/person/1")
    assert p_resp.status_code == 200
    p_data = p_resp.json()
    assert p_data["person_id"] == 1
    assert p_data["expression_timeline"] == ["Neutral", "Happy", "Surprise"]
    assert "dominant_expression" in p_data

def test_session_history_endpoint():
    """Test 7: GET /api/sessions."""
    response = client.get("/api/sessions?page=1&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "sessions" in data
    assert isinstance(data["sessions"], list)

def test_session_details_endpoint():
    """Test 8: GET /api/sessions/{session_id}."""
    start_resp = client.post("/api/session/start", json={"session_name": "Session Details Test"})
    session_id = start_resp.json()["session_id"]
    
    details_resp = client.get(f"/api/sessions/{session_id}")
    assert details_resp.status_code == 200
    data = details_resp.json()
    assert data["session_id"] == session_id
    assert data["session_name"] == "Session Details Test"

def test_error_handling_invalid_session():
    """Test 9: Invalid session ID returns 404 Not Found."""
    resp = client.get("/api/session/invalid_session_id_999/analytics")
    assert resp.status_code == 404
    assert "detail" in resp.json()

def test_error_handling_invalid_person():
    """Test 10: Non-existent person ID returns 404 Not Found."""
    start_resp = client.post("/api/session/start", json={"session_name": "Invalid Person Test"})
    session_id = start_resp.json()["session_id"]
    
    resp = client.get(f"/api/session/{session_id}/person/9999")
    assert resp.status_code == 404
    assert "detail" in resp.json()
