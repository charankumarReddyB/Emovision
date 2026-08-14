"""
Health and Model Information API Endpoints.
"""
from fastapi import APIRouter
from datetime import datetime
from app.core.config import settings
from app.schemas.api_schemas import HealthResponse, ModelInfoResponse

router = APIRouter(tags=["Health & Information"])

@router.get("/api/health", response_model=HealthResponse)
def get_health_status():
    """
    Health check endpoint returning system, CV model, tracking status, and current timestamp.
    """
    return {
        "status": "ok",
        "cv_model_status": "ready",
        "tracking_status": "active",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/api/model", response_model=ModelInfoResponse)
@router.get("/api/model/info", response_model=ModelInfoResponse)
def get_model_information():
    """
    Model information endpoint returning model specifications, 7 target emotion classes, and input shape.
    """
    return {
        "model_name": "EmotionCNN (Lightweight 4-block ConvNet)",
        "emotion_classes": settings.EMOTION_CLASSES,
        "model_status": "ready",
        "input_face_size": list(settings.TARGET_FACE_SIZE)
    }
