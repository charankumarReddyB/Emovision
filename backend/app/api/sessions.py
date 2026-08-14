"""
Session Management API Endpoints.
Handles session start/end lifecycle, current real-time detection results, session history, and details.
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from datetime import datetime
import uuid
from typing import List
from collections import Counter

from app.db.repository import get_db_repository, DatabaseRepository
from app.schemas.api_schemas import (
    SessionStartRequest,
    SessionStartResponse,
    SessionEndResponse,
    CurrentDetectionResponse,
    SessionHistoryResponse
)

router = APIRouter(tags=["Sessions"])

def get_repo() -> DatabaseRepository:
    return get_db_repository()

@router.post("/api/session/start", response_model=SessionStartResponse, status_code=status.HTTP_201_CREATED)
def start_session(payload: SessionStartRequest, repo: DatabaseRepository = Depends(get_repo)):
    """
    Creates and starts a new tracking session.
    """
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    result = repo.create_session(
        session_id=session_id,
        session_name=payload.session_name or "Live Session",
        source_type=payload.source_type or "webcam"
    )
    return {
        "session_id": result["session_id"],
        "start_time": result.get("start_time", datetime.utcnow().isoformat()),
        "status": result.get("status", "active")
    }

@router.post("/api/session/{session_id}/end", response_model=SessionEndResponse)
def end_session(session_id: str, repo: DatabaseRepository = Depends(get_repo)):
    """
    Finalizes an active session, computes overall statistics, and updates session record.
    """
    result = repo.end_session(session_id=session_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Session ID '{session_id}' not found.")
    return result

@router.get("/api/session/{session_id}/current", response_model=CurrentDetectionResponse)
def get_current_detection(session_id: str, repo: DatabaseRepository = Depends(get_repo)):
    """
    Returns the latest real-time structured detection results for a session.
    """
    res = repo.get_latest_frame_detections(session_id=session_id)
    if not res:
        raise HTTPException(status_code=404, detail=f"Session ID '{session_id}' not found.")
    return res

@router.get("/api/sessions", response_model=SessionHistoryResponse)
@router.get("/api/sessions/history", response_model=SessionHistoryResponse)
def get_session_history(
    page: int = Query(1, ge=1, description="Page index"),
    limit: int = Query(10, ge=1, le=50, description="Items per page"),
    repo: DatabaseRepository = Depends(get_repo)
):
    """
    Returns previous tracking sessions with pagination.
    """
    return repo.list_sessions(page=page, limit=limit)

@router.get("/api/sessions/{session_id}")
@router.get("/api/session/{session_id}/details")
def get_session_details(session_id: str, repo: DatabaseRepository = Depends(get_repo)):
    """
    Returns complete detailed stored information for a selected session.
    """
    sess = repo.get_session(session_id=session_id)
    if not sess:
        raise HTTPException(status_code=404, detail=f"Session ID '{session_id}' not found.")
    return sess
