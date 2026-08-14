"""
SQLAlchemy Database ORM Models for Emovision Backend.
Defines tables for tracking sessions and structured per-frame face detections.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()

class SessionModel(Base):
    """
    Table storing camera, video, or synthetic tracking sessions.
    """
    __tablename__ = "sessions"

    session_id = Column(String, primary_key=True, index=True)
    session_name = Column(String, nullable=False, default="Live Session")
    source_type = Column(String, nullable=False, default="webcam")
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    total_frames = Column(Integer, default=0)
    total_predictions = Column(Integer, default=0)
    total_people_detected = Column(Integer, default=0)
    avg_fps = Column(Float, default=0.0)
    avg_confidence = Column(Float, default=0.0)
    dominant_expression = Column(String, nullable=True)
    status = Column(String, default="active")

    @property
    def duration(self) -> float:
        """Calculates elapsed session duration in seconds."""
        if self.start_time and self.end_time:
            return round((self.end_time - self.start_time).total_seconds(), 1)
        elif self.start_time:
            return round((datetime.utcnow() - self.start_time).total_seconds(), 1)
        return 0.0

    # Relationship to detection logs
    detections = relationship("DetectionLogModel", back_populates="session", cascade="all, delete-orphan")

class DetectionLogModel(Base):
    """
    Table storing per-frame face detection metrics (Person ID, bounding box, expression, confidence).
    """
    __tablename__ = "face_detections"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    session_id = Column(String, ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    frame_number = Column(Integer, nullable=False)
    person_id = Column(Integer, nullable=False, index=True)
    bbox_x = Column(Integer, nullable=False)
    bbox_y = Column(Integer, nullable=False)
    bbox_w = Column(Integer, nullable=False)
    bbox_h = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False, default=1.0)
    emotion_label = Column(String, nullable=True)
    emotion_confidence = Column(Float, nullable=True, default=0.0)
    timestamp = Column(DateTime, default=datetime.utcnow)

    session = relationship("SessionModel", back_populates="detections")
