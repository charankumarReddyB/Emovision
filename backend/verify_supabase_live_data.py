"""
Supabase Production Database Verification Script for Emovision Platform.
Performs an actual live detection session against Supabase PostgreSQL and audits
session rows, prediction rows, field integrity, and analytics calculation consistency.
"""
import os
import sys
import uuid
import time
from datetime import datetime
from collections import Counter

from app.core.config import settings
from app.db.repository import get_db_repository, SupabaseRepository

def verify_live_supabase_data():
    print("==========================================================================")
    print("      EMOVISION LIVE SUPABASE POSTGRESQL DATA VERIFICATION AUDIT          ")
    print("==========================================================================")

    repo = get_db_repository()
    repo_name = repo.__class__.__name__
    print(f"[Database Layer] Active Repository: {repo_name}")
    print(f"[Supabase URL] {settings.SUPABASE_URL}")

    if not isinstance(repo, SupabaseRepository) or not repo.client:
        print("[FAIL] Active repository is NOT connected to Supabase PostgreSQL!")
        return False

    session_id = f"audit_sess_{uuid.uuid4().hex[:8]}"
    print(f"\n--- 1. Starting Live Session: {session_id} ---")
    start_res = repo.create_session(session_id=session_id, session_name="Production Verification Session", source_type="live_test")
    print(f"[Success] Session Created: {start_res}")

    print("\n--- 2. Simulating Multi-Person Prediction Frames ---")
    # Simulate 3 consecutive frames with 3 tracked people (P1, P2, P3)
    frame_1_dets = [
        {"person_id": 1, "bbox": (100, 100, 60, 60), "confidence": 0.95, "emotion": "Happy", "emotion_confidence": 0.92},
        {"person_id": 2, "bbox": (250, 120, 55, 55), "confidence": 0.89, "emotion": "Neutral", "emotion_confidence": 0.85},
        {"person_id": 3, "bbox": (400, 110, 65, 65), "confidence": 0.91, "emotion": "Surprise", "emotion_confidence": 0.88}
    ]
    frame_2_dets = [
        {"person_id": 1, "bbox": (102, 101, 60, 60), "confidence": 0.96, "emotion": "Happy", "emotion_confidence": 0.94},
        {"person_id": 2, "bbox": (252, 121, 55, 55), "confidence": 0.90, "emotion": "Happy", "emotion_confidence": 0.82},
        {"person_id": 3, "bbox": (398, 112, 65, 65), "confidence": 0.92, "emotion": "Surprise", "emotion_confidence": 0.91}
    ]

    repo.log_frame_predictions(session_id, frame_number=1, detections=frame_1_dets)
    repo.log_frame_predictions(session_id, frame_number=2, detections=frame_2_dets)
    
    # Short sleep to allow background thread worker tasks to complete
    time.sleep(1.5)

    print("\n--- 3. Finalizing Session ---")
    end_res = repo.end_session(session_id=session_id, total_frames=2, avg_fps=30.0)
    print(f"[Success] Session Finalized: {end_res}")

    print("\n--- 4. Direct Supabase Table Audit ---")
    client = repo.client

    # Query Sessions Table
    sess_row_res = client.table("sessions").select("*").eq("id", session_id).execute()
    assert len(sess_row_res.data) == 1, "Session row not found in Supabase!"
    sess_row = sess_row_res.data[0]

    # Query Predictions Table
    preds_row_res = client.table("predictions").select("*").eq("session_id", session_id).order("frame_number").execute()
    preds_rows = preds_row_res.data or []

    print(f"[Supabase Session Row] ID: {sess_row['id']}, Status: {sess_row['status']}, People: {sess_row['people_count']}, Predictions: {sess_row['total_predictions']}, Dominant: {sess_row['dominant_expression']}")
    print(f"[Supabase Prediction Rows] Created Row Count: {len(preds_rows)}")

    # Verify Predictions Fields
    assert len(preds_rows) == 6, f"Expected 6 prediction rows, found {len(preds_rows)}"
    pids_found = set(p["person_id"] for p in preds_rows)
    emotions_found = set(p["expression"] for p in preds_rows)
    assert pids_found == {1, 2, 3}, f"Unexpected Person IDs: {pids_found}"
    assert "Happy" in emotions_found and "Surprise" in emotions_found, f"Unexpected emotions: {emotions_found}"

    for p in preds_rows:
        assert p["session_id"] == session_id
        assert p["person_id"] in [1, 2, 3]
        assert p["x"] > 0 and p["y"] > 0 and p["width"] > 0 and p["height"] > 0
        assert p["confidence"] > 0.0
        assert p["timestamp"] is not None

    print("\n--- 5. Analytics Calculation Discrepancy Check ---")
    analytics = repo.get_session_analytics(session_id)
    assert analytics["total_predictions"] == len(preds_rows) == 6, "Prediction count discrepancy!"
    assert analytics["total_people_detected"] == len(pids_found) == 3, "People count discrepancy!"
    assert analytics["dominant_expression"] == "Happy", f"Dominant emotion calculation mismatch! Expected Happy, got {analytics['dominant_expression']}"

    person_1_analytics = repo.get_person_analytics(session_id, person_id=1)
    assert person_1_analytics["person_id"] == 1
    assert person_1_analytics["expression_timeline"] == ["Happy", "Happy"]
    assert person_1_analytics["dominant_expression"] == "Happy"

    print("[PASS] Zero calculation discrepancies detected! Supabase PostgreSQL data persistence verified 100%.")
    return True

if __name__ == "__main__":
    success = verify_live_supabase_data()
    sys.exit(0 if success else 1)
