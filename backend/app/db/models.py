"""
Pydantic schemas for API requests, responses, and CV detection data structures.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

class BoundingBox(BaseModel):
    x: int = Field(..., description="Top-left X coordinate")
    y: int = Field(..., description="Top-left Y coordinate")
    width: int = Field(..., description="Bounding box width")
    height: int = Field(..., description="Bounding box height")

class DetectedFace(BaseModel):
    person_id: Optional[int] = Field(None, description="Assigned persistent tracker Person ID")
    bbox: Tuple[int, int, int, int] = Field(..., description="(x, y, w, h) bounding box")
    confidence: float = Field(..., description="Face detection confidence score")
    emotion: Optional[str] = Field(None, description="Classified emotion label")
    emotion_confidence: Optional[float] = Field(None, description="Emotion prediction confidence")

class FrameAnalysisResult(BaseModel):
    frame_number: int
    detected_faces_count: int
    faces: List[DetectedFace]
    fps: float
    timestamp_ms: float

class SessionCreateRequest(BaseModel):
    session_name: str = Field(..., description="Descriptive name for the session")
    source_type: str = Field("webcam", description="Source: 'webcam', 'video', or 'image'")

class SessionResponse(BaseModel):
    session_id: str
    session_name: str
    source_type: str
    start_time: str
    status: str
