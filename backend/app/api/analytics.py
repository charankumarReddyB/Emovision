"""
Analytics API Endpoints.
Handles session-wide aggregate metrics and per-person analytics with expression timeline tracking.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from collections import Counter

from app.db.repository import get_db_repository, DatabaseRepository
from app.schemas.api_schemas import SessionAnalyticsResponse, PersonAnalyticsResponse

router = APIRouter(tags=["Analytics"])

def get_repo() -> DatabaseRepository:
    return get_db_repository()

@router.get("/api/session/{session_id}/analytics", response_model=SessionAnalyticsResponse)
def get_session_analytics(session_id: str, repo: DatabaseRepository = Depends(get_repo)):
    """
    Returns session-wide aggregate analytics including emotion breakdown, average confidence, and FPS stats.
    """
    res = repo.get_session_analytics(session_id=session_id)
    if not res:
        raise HTTPException(status_code=404, detail=f"Session ID '{session_id}' not found.")
    return res

@router.get("/api/session/{session_id}/person/{person_id}", response_model=PersonAnalyticsResponse)
def get_person_analytics(session_id: str, person_id: int, repo: DatabaseRepository = Depends(get_repo)):
    """
    Returns detailed analytics for a specific tracked Person ID within a session,
    including dominant emotion, confidence, distribution, and chronological expression timeline.
    """
    # Check if session exists
    sess = repo.get_session(session_id=session_id)
    if not sess:
        raise HTTPException(status_code=404, detail=f"Session ID '{session_id}' not found.")

    res = repo.get_person_analytics(session_id=session_id, person_id=person_id)
    if not res:
        raise HTTPException(
            status_code=404,
            detail=f"Person ID '{person_id}' not found in session '{session_id}'."
        )
    return res
