"""
Session & Real-Time Statistics Collector for Emovision.
Calculates live session statistics (Dominant Emotion, Per-Expression Counts, Average Confidence, Duration)
and logs structured prediction metrics to SQLite database.
"""
import time
from typing import List, Dict, Any
from collections import Counter
from app.db.database import create_session, log_frame_detections, close_session

class SessionTracker:
    """
    Tracks live statistics and records session data into SQLite.
    """
    def __init__(self, session_id: str, session_name: str = "Live Session", source_type: str = "webcam"):
        self.session_id = session_id
        self.session_name = session_name
        self.source_type = source_type
        self.start_time = time.time()
        self.total_frames = 0
        
        # Initialize SQLite session
        create_session(self.session_id, self.session_name, self.source_type)

    def process_frame_detections(self, frame_number: int, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Processes current frame detections, updates SQLite database logs, and returns live statistics.
        
        Args:
            frame_number (int): Frame index.
            detections (List[Dict[str, Any]]): List of detected/tracked faces with 'emotion' and 'emotion_confidence'.
            
        Returns:
            Dict[str, Any]: Live statistics summary dictionary.
        """
        self.total_frames = frame_number
        
        # Log structured prediction data to SQLite
        log_frame_detections(self.session_id, frame_number, detections)
        
        total_people = len(detections)
        if total_people == 0:
            return {
                "total_people": 0,
                "expression_counts": {},
                "dominant_expression": "None",
                "average_confidence": 0.0,
                "session_duration_sec": round(time.time() - self.start_time, 1)
            }
            
        emotions = [d.get("emotion", "Neutral") for d in detections]
        confidences = [d.get("emotion_confidence", 0.0) for d in detections]
        
        counts = dict(Counter(emotions))
        dominant_emotion = max(counts, key=counts.get)
        avg_confidence = float(sum(confidences) / len(confidences)) if confidences else 0.0
        
        return {
            "total_people": total_people,
            "expression_counts": counts,
            "dominant_expression": dominant_emotion,
            "average_confidence": round(avg_confidence * 100, 1),
            "session_duration_sec": round(time.time() - self.start_time, 1)
        }

    def finish_session(self, avg_fps: float):
        """Closes session in SQLite database."""
        close_session(self.session_id, self.total_frames, avg_fps)
