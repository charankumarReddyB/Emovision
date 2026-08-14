"""
Database Access Layer for Emovision Backend.
Integrates SQLAlchemy Engine & Session Factory for SQLite storage.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import sqlite3
from typing import Generator, Dict, Any, List
from app.core.config import settings
from app.db.orm_models import Base, SessionModel, DetectionLogModel
from app.db.repository import get_db_repository

# SQLAlchemy Engine setup
SQLALCHEMY_DATABASE_URL = f"sqlite:///{settings.DATABASE_PATH}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db_connection():
    """Raw SQLite connection for lightweight querying."""
    conn = sqlite3.connect(str(settings.DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def get_db() -> Generator[Session, None, None]:
    """FastAPI Dependency providing a database ORM session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Creates database tables if they do not exist, ensuring column migrations."""
    # Check if existing table needs columns
    conn = sqlite3.connect(str(settings.DATABASE_PATH))
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA table_info(sessions)")
        cols = [r[1] for r in cursor.fetchall()]
        if cols and "total_predictions" not in cols:
            conn.close()
            # Drop old incompatible schema tables so SQLAlchemy can recreate them cleanly
            conn = sqlite3.connect(str(settings.DATABASE_PATH))
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS face_detections")
            cursor.execute("DROP TABLE IF EXISTS sessions")
            conn.commit()
    except Exception:
        pass
    finally:
        conn.close()

    Base.metadata.create_all(bind=engine)

def create_session(session_id: str, session_name: str, source_type: str) -> Dict[str, Any]:
    """Delegates session creation to active database repository (Supabase/SQLite)."""
    repo = get_db_repository()
    return repo.create_session(session_id=session_id, session_name=session_name, source_type=source_type)

def log_frame_detections(session_id: str, frame_number: int, detections: List[Dict[str, Any]]):
    """Delegates frame detection logging to active database repository (Supabase/SQLite)."""
    repo = get_db_repository()
    return repo.log_frame_predictions(session_id=session_id, frame_number=frame_number, detections=detections)

def close_session(session_id: str, total_frames: int, avg_fps: float):
    """Delegates session finalization to active database repository (Supabase/SQLite)."""
    repo = get_db_repository()
    return repo.end_session(session_id=session_id, total_frames=total_frames, avg_fps=avg_fps)
