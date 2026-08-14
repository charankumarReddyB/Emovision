"""
FastAPI Main Application Entrypoint for Emovision Backend.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from app.core.config import settings
from app.db.database import init_db, create_session, get_db_connection
from app.db.models import SessionCreateRequest, SessionResponse

# Initialize SQLite tables
init_db()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Backend API for Emovision Real-Time Human Emotion Recognition & Face Tracking"
)

# Enable CORS for communication with frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "target_emotions": settings.EMOTION_CLASSES
    }

@app.get("/health")
def health_check():
    """Health check endpoint confirming database and server readiness."""
    try:
        conn = get_db_connection()
        conn.execute("SELECT 1")
        conn.close()
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
        
    return {
        "status": "ok",
        "database": db_status,
        "detector_type": settings.DETECTOR_TYPE,
        "target_face_size": settings.TARGET_FACE_SIZE
    }

@app.post(f"{settings.API_PREFIX}/sessions", response_model=SessionResponse)
def start_session(payload: SessionCreateRequest):
    """Creates a new tracking session."""
    session_id = str(uuid.uuid4())[:8]
    res = create_session(session_id, payload.session_name, payload.source_type)
    return {
        "session_id": session_id,
        "session_name": payload.session_name,
        "source_type": payload.source_type,
        "start_time": str(os.getenv("CURRENT_TIME", "")),
        "status": "active"
    }

@app.get(f"{settings.API_PREFIX}/sessions")
def list_sessions():
    """Returns recent sessions logged in SQLite."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions ORDER BY start_time DESC LIMIT 20")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return {"sessions": rows}
