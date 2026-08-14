"""
SQLite Database Layer for Emovision.
Manages database initialization, connections, session storage, and analytics logging.
"""
import sqlite3
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.core.config import settings

def get_db_connection() -> sqlite3.Connection:
    """Returns a SQLite connection with dict-like row access."""
    conn = sqlite3.connect(str(settings.DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes SQLite tables for sessions, detections, and analytics."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Sessions Table (Live webcam session, video file session, image session)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            session_name TEXT NOT NULL,
            source_type TEXT NOT NULL, -- 'webcam', 'video', 'image'
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP,
            total_frames INTEGER DEFAULT 0,
            avg_fps REAL DEFAULT 0.0,
            status TEXT DEFAULT 'active'
        );
    """)
    
    # Detections Log Table (Per-frame detected face logs with Person ID and bounding box)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            frame_number INTEGER NOT NULL,
            person_id INTEGER NOT NULL,
            bbox_x INTEGER NOT NULL,
            bbox_y INTEGER NOT NULL,
            bbox_w INTEGER NOT NULL,
            bbox_h INTEGER NOT NULL,
            confidence REAL NOT NULL,
            emotion_label TEXT,
            emotion_confidence REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
        );
    """)
    
    # Analytics Summary Table (Aggregated counts per session/person)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analytics_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            person_id INTEGER NOT NULL,
            total_frames_tracked INTEGER DEFAULT 0,
            dominant_emotion TEXT,
            emotion_breakdown_json TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
        );
    """)
    
    conn.commit()
    conn.close()

def create_session(session_id: str, session_name: str, source_type: str) -> Dict[str, Any]:
    """Creates a new tracking session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO sessions (session_id, session_name, source_type)
        VALUES (?, ?, ?)
    """, (session_id, session_name, source_type))
    conn.commit()
    conn.close()
    return {"session_id": session_id, "session_name": session_name, "source_type": source_type}

def log_frame_detections(session_id: str, frame_number: int, detections: List[Dict[str, Any]]):
    """Logs a list of detected & tracked faces in a frame."""
    conn = get_db_connection()
    cursor = conn.cursor()
    for det in detections:
        bbox = det.get("bbox", (0, 0, 0, 0))
        cursor.execute("""
            INSERT INTO face_detections 
            (session_id, frame_number, person_id, bbox_x, bbox_y, bbox_w, bbox_h, confidence, emotion_label, emotion_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            frame_number,
            det.get("person_id", -1),
            bbox[0], bbox[1], bbox[2], bbox[3],
            det.get("confidence", 1.0),
            det.get("emotion", None),
            det.get("emotion_confidence", 0.0)
        ))
    conn.commit()
    conn.close()

def close_session(session_id: str, total_frames: int, avg_fps: float):
    """Closes an active session and computes aggregate stats."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE sessions
        SET end_time = CURRENT_TIMESTAMP,
            total_frames = ?,
            avg_fps = ?,
            status = 'completed'
        WHERE session_id = ?
    """, (total_frames, avg_fps, session_id))
    conn.commit()
    conn.close()
