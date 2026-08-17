"""
Pydantic API Request and Response Schemas for Emovision Backend.
Defines structured data models for health, model info, sessions, analytics, person identification cards, and detections.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime

# 1. Health & Model Info Schemas
class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    cv_model_status: str = Field(..., example="ready")
    tracking_status: str = Field(..., example="active")
    timestamp: str = Field(..., example="2026-08-14T15:38:42")

class ModelInfoResponse(BaseModel):
    model_name: str = Field(..., example="EmotionCNN (Lightweight ConvNet)")
    emotion_classes: List[str] = Field(..., example=["Happy", "Sad", "Angry", "Fear", "Surprise", "Disgust", "Neutral"])
    model_status: str = Field(..., example="ready")
    input_face_size: List[int] = Field(..., example=[48, 48])

# 2. Session Lifecycle Schemas
class SessionStartRequest(BaseModel):
    session_name: Optional[str] = Field("Live Session", description="Descriptive session title")
    source_type: Optional[str] = Field("webcam", description="Source type: 'webcam', 'video', 'synthetic'")

class SessionStartResponse(BaseModel):
    session_id: str = Field(..., example="sess_a1b2c3")
    start_time: str = Field(..., example="2026-08-14T15:38:42")
    status: str = Field(..., example="active")

class SessionEndResponse(BaseModel):
    session_id: str = Field(..., example="sess_a1b2c3")
    start_time: str = Field(..., example="2026-08-14T15:38:42")
    end_time: str = Field(..., example="2026-08-14T15:40:12")
    duration_seconds: float = Field(..., example=90.0)
    total_predictions: int = Field(..., example=450)
    total_people_detected: int = Field(..., example=3)
    dominant_expression: str = Field(..., example="Happy")

# 3. Person Identification Schemas
class PersonDetailCard(BaseModel):
    person_id: int = Field(..., example=1)
    thumbnail_b64: Optional[str] = Field("", description="Base64 encoded JPEG face crop thumbnail URL")
    dominant_emotion: str = Field(..., example="Happy")
    average_confidence: float = Field(..., example=92.5)
    total_detections: int = Field(..., example=120)

# 4. Detection Schemas
class BoundingBoxSchema(BaseModel):
    x: int = Field(..., example=120)
    y: int = Field(..., example=80)
    width: int = Field(..., example=150)
    height: int = Field(..., example=160)

class PersonDetectionSchema(BaseModel):
    person_id: int = Field(..., example=1)
    expression: str = Field(..., example="Happy")
    confidence: float = Field(..., example=0.92)
    bounding_box: BoundingBoxSchema

class CurrentDetectionResponse(BaseModel):
    session_id: str = Field(..., example="sess_a1b2c3")
    people_detected: int = Field(..., example=2)
    fps: float = Field(..., example=28.5)
    average_confidence: float = Field(..., example=88.5)
    dominant_expression: str = Field(..., example="Happy")
    people: List[PersonDetectionSchema]

# 5. Analytics Schemas
class SessionAnalyticsResponse(BaseModel):
    session_id: str = Field(..., json_schema_extra={"example": "sess_a1b2c3"})
    total_people_detected: int = Field(..., json_schema_extra={"example": 4})
    total_predictions: int = Field(..., json_schema_extra={"example": 1200})
    expression_distribution: Dict[str, int] = Field(..., json_schema_extra={"example": {"Happy": 600, "Neutral": 400, "Surprise": 200}})
    average_confidence: float = Field(..., json_schema_extra={"example": 86.4})
    dominant_expression: str = Field(..., json_schema_extra={"example": "Happy"})
    session_duration_seconds: float = Field(..., json_schema_extra={"example": 120.5})
    persons: List[int] = Field(default_factory=list)
    persons_details: List[PersonDetailCard] = Field(default_factory=list)
    avg_fps: float = Field(..., example=30.2)

class PersonAnalyticsResponse(BaseModel):
    person_id: int = Field(..., example=1)
    thumbnail_b64: Optional[str] = Field("", description="Base64 encoded JPEG face crop thumbnail")
    dominant_expression: str = Field(..., example="Happy")
    average_confidence: float = Field(..., example=91.2)
    expression_distribution: Dict[str, int] = Field(..., example={"Happy": 85, "Neutral": 15})
    expression_timeline: List[str] = Field(..., example=["Neutral", "Neutral", "Happy", "Happy", "Surprise", "Neutral"])

# 6. History Schemas
class SessionSummarySchema(BaseModel):
    session_id: str = Field(..., example="sess_a1b2c3")
    session_name: str = Field(..., example="Live Webcam Session")
    date: str = Field(..., example="2026-08-14")
    duration_seconds: float = Field(..., example=120.0)
    people_count: int = Field(..., example=3)
    dominant_expression: str = Field(..., example="Happy")
    average_confidence: float = Field(..., example=89.5)
    status: str = Field(..., example="completed")
    persons_details: List[PersonDetailCard] = Field(default_factory=list)

class SessionHistoryResponse(BaseModel):
    total: int = Field(..., example=15)
    page: int = Field(..., example=1)
    limit: int = Field(..., example=10)
    sessions: List[SessionSummarySchema]
