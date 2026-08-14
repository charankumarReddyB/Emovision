"""
Database Migration Script: SQLite to Supabase PostgreSQL for Emovision Platform.
Safely reads session records and face detection logs from SQLite (data/emovision.db)
and copies them into Supabase PostgreSQL tables (sessions, predictions).
"""
import os
import sys
import sqlite3
import argparse
from datetime import datetime
from typing import List, Dict, Any

from app.core.config import settings
from app.db.repository import SupabaseRepository, SqliteRepository

def migrate_sqlite_to_supabase(verify_only: bool = False, batch_size: int = 100):
    print("==========================================================================")
    print("      EMOVISION DATABASE MIGRATION: SQLITE -> SUPABASE POSTGRESQL         ")
    print("==========================================================================")

    db_path = settings.DATABASE_PATH
    if not os.path.exists(db_path):
        print(f"[ERROR] SQLite database file not found at: {db_path}")
        return False

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 1. Read SQLite Sessions
    cursor.execute("SELECT * FROM sessions")
    sqlite_sessions = cursor.fetchall()
    
    # 2. Read SQLite Detections
    cursor.execute("SELECT * FROM face_detections")
    sqlite_detections = cursor.fetchall()

    print(f"[SQLite Audit] Total Sessions Stored: {len(sqlite_sessions)}")
    print(f"[SQLite Audit] Total Frame Detections Stored: {len(sqlite_detections)}")

    if verify_only:
        print("[Verify Only] Migration audit complete. No records modified.")
        conn.close()
        return True

    # Check Supabase connection credentials
    if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
        print("[WARNING] SUPABASE_URL or SUPABASE_KEY not set in environment or config.")
        print("Please configure SUPABASE_URL and SUPABASE_KEY in your .env file or environment.")
        conn.close()
        return False

    supabase_repo = SupabaseRepository(url=settings.SUPABASE_URL, key=settings.SUPABASE_KEY)
    if not supabase_repo.client:
        print("[ERROR] Failed to initialize Supabase client. Check credentials.")
        conn.close()
        return False

    print("\n--- 1. Migrating Sessions to Supabase ---")
    migrated_sessions = 0
    for s in sqlite_sessions:
        try:
            start_iso = s["start_time"] if s["start_time"] else datetime.utcnow().isoformat()
            end_iso = s["end_time"] if s["end_time"] else None
            sess_payload = {
                "id": s["session_id"],
                "session_name": s["session_name"] or "Live Session",
                "source_type": s["source_type"] or "webcam",
                "started_at": start_iso,
                "ended_at": end_iso,
                "duration": float(s["total_frames"] / max(s["avg_fps"], 1.0)) if ("total_frames" in s.keys() and s["total_frames"] and s["avg_fps"]) else 0.0,
                "people_count": s["total_people_detected"] or 0,
                "total_predictions": s["total_predictions"] or 0,
                "dominant_expression": s["dominant_expression"] or "Neutral",
                "average_confidence": float(s["avg_confidence"] or 0.0),
                "status": s["status"] or "completed"
            }
            supabase_repo.client.table("sessions").upsert(sess_payload).execute()
            migrated_sessions += 1
        except Exception as e:
            print(f"[Warning] Failed to migrate session '{s['session_id']}': {e}")

    print(f"[Success] Migrated {migrated_sessions} / {len(sqlite_sessions)} sessions to Supabase.")

    print("\n--- 2. Migrating Predictions to Supabase ---")
    migrated_predictions = 0
    batch = []
    for d in sqlite_detections:
        ts_iso = d["timestamp"] if d["timestamp"] else datetime.utcnow().isoformat()
        pred_payload = {
            "session_id": d["session_id"],
            "person_id": d["person_id"],
            "frame_number": d["frame_number"],
            "timestamp": ts_iso,
            "expression": d["emotion_label"] or "Neutral",
            "confidence": float(d["emotion_confidence"] or 0.0),
            "x": d["bbox_x"],
            "y": d["bbox_y"],
            "width": d["bbox_w"],
            "height": d["bbox_h"]
        }
        batch.append(pred_payload)

        if len(batch) >= batch_size:
            try:
                supabase_repo.client.table("predictions").insert(batch).execute()
                migrated_predictions += len(batch)
            except Exception as e:
                print(f"[Warning] Batch insert error: {e}")
            batch = []

    if batch:
        try:
            supabase_repo.client.table("predictions").insert(batch).execute()
            migrated_predictions += len(batch)
        except Exception as e:
            print(f"[Warning] Final batch insert error: {e}")

    print(f"[Success] Migrated {migrated_predictions} / {len(sqlite_detections)} predictions to Supabase.")

    # Verification Step
    print("\n--- 3. Post-Migration Record Count Verification ---")
    supa_sess_count = len(supabase_repo.client.table("sessions").select("id").execute().data or [])
    supa_pred_count = supabase_repo.client.table("predictions").select("id", count="exact").execute().count or 0

    print(f"SQLite Sessions: {len(sqlite_sessions)} | Supabase Sessions: {supa_sess_count}")
    print(f"SQLite Predictions: {len(sqlite_detections)} | Supabase Predictions: {supa_pred_count}")
    print(f"Local SQLite Backup Preserved at: {db_path}")

    conn.close()
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emovision SQLite to Supabase Database Migration Tool")
    parser.add_argument("--verify-only", action="store_true", help="Audit records without inserting to Supabase")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch insert size for predictions")
    args = parser.parse_args()

    success = migrate_sqlite_to_supabase(verify_only=args.verify_only, batch_size=args.batch_size)
    sys.exit(0 if success else 1)
