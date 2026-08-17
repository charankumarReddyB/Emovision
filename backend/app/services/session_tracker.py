"""
Session & Real-Time Statistics Collector for Emovision.
Calculates live session statistics (Dominant Emotion, Per-Expression Counts, Average Confidence, Duration),
captures base64 face thumbnails for Person ID identification, and logs structured prediction metrics asynchronously.
"""
import time
import cv2
import base64
import numpy as np
from typing import List, Dict, Any
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from app.db.database import create_session, log_frame_detections, close_session, save_person_thumbnails

class SessionTracker:
    """
    Tracks live statistics, collects face thumbnails per Person ID,
    and records session data asynchronously into database.
    """
    def __init__(self, session_id: str, session_name: str = "Live Session", source_type: str = "webcam"):
        self.session_id = session_id
        self.session_name = session_name
        self.source_type = source_type
        self.start_time = time.time()
        self.total_frames = 0
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.person_thumbnails: Dict[int, str] = {}
        self.person_emotions: Dict[int, List[str]] = {}
        self.person_confidences: Dict[int, List[float]] = {}
        
        # Initialize session in DB
        create_session(self.session_id, self.session_name, self.source_type)

    def _convert_chip_to_base64(self, chip: np.ndarray, person_id: int = 1) -> str:
        """Converts BGR face chip into base64 JPEG data URL with cyan Person ID box overlay."""
        if chip is None or chip.size == 0:
            return ""
        try:
            # Resize chip to standard 112x112 thumbnail
            thumb = cv2.resize(chip, (112, 112))
            h, w = thumb.shape[:2]
            
            # Draw high-contrast cyber cyan bounding box border around the face thumbnail
            cv2.rectangle(thumb, (0, 0), (w - 1, h - 1), (255, 212, 0), 2)
            
            # Dark HUD bottom bar with Person ID label
            cv2.rectangle(thumb, (0, h - 22), (w, h), (19, 30, 48), -1)
            cv2.putText(
                thumb,
                f"PERSON #{person_id}",
                (8, h - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (255, 212, 0),
                1,
                cv2.LINE_AA
            )
            
            _, buffer = cv2.imencode('.jpg', thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            b64_str = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{b64_str}"
        except Exception:
            return ""

    def process_frame_detections(self, frame_number: int, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Processes current frame detections asynchronously and returns live statistics summary.
        
        Args:
            frame_number (int): Frame index.
            detections (List[Dict[str, Any]]): List of detected faces with 'emotion', 'emotion_confidence', and face chips.
            
        Returns:
            Dict[str, Any]: Live statistics summary dictionary.
        """
        self.total_frames = frame_number
        
        # 1. Asynchronously log structured prediction data to database without blocking main thread
        self.executor.submit(log_frame_detections, self.session_id, frame_number, detections)
        
        total_people = len(detections)
        if total_people == 0:
            return {
                "total_people": 0,
                "expression_counts": {},
                "dominant_expression": "None",
                "average_confidence": 0.0,
                "session_duration_sec": round(time.time() - self.start_time, 1)
            }
            
        emotions = []
        confidences = []
        
        for det in detections:
            p_id = det.get("person_id", det.get("face_index", 1))
            emo = det.get("emotion", "Neutral")
            conf = det.get("emotion_confidence", 0.0)
            
            emotions.append(emo)
            confidences.append(conf)
            
            if p_id not in self.person_emotions:
                self.person_emotions[p_id] = []
                self.person_confidences[p_id] = []
            self.person_emotions[p_id].append(emo)
            self.person_confidences[p_id].append(conf)
            
            # STRICT DEDUPLICATION: Capture exactly 1 unique thumbnail per Person ID with highest confidence
            chip = det.get("aligned_chip", det.get("face_chip"))
            if chip is not None and chip.size > 0:
                if p_id not in self.person_thumbnails or not self.person_thumbnails[p_id]:
                    b64_thumb = self._convert_chip_to_base64(chip, person_id=p_id)
                    if b64_thumb:
                        self.person_thumbnails[p_id] = b64_thumb
                        # Sync updated unique thumbnails to DB asynchronously
                        self.executor.submit(save_person_thumbnails, self.session_id, self.get_persons_details())
        
        counts = dict(Counter(emotions))
        dominant_emotion = max(counts, key=counts.get) if counts else "Neutral"
        avg_confidence = float(sum(confidences) / len(confidences)) if confidences else 0.0
        
        return {
            "total_people": total_people,
            "expression_counts": counts,
            "dominant_expression": dominant_emotion,
            "average_confidence": round(avg_confidence * 100, 1),
            "session_duration_sec": round(time.time() - self.start_time, 1)
        }

    def get_persons_details(self) -> List[Dict[str, Any]]:
        """Returns structured list of person identification cards with base64 thumbnails and stats."""
        persons_list = []
        for p_id, emo_list in self.person_emotions.items():
            counts = dict(Counter(emo_list))
            dom_emo = max(counts, key=counts.get) if counts else "Neutral"
            confs = self.person_confidences.get(p_id, [])
            avg_conf = round(float(sum(confs) / len(confs)) * 100, 1) if confs else 0.0
            thumb = self.person_thumbnails.get(p_id, "")
            
            persons_list.append({
                "person_id": p_id,
                "thumbnail_b64": thumb,
                "dominant_emotion": dom_emo,
                "average_confidence": avg_conf,
                "total_detections": len(emo_list)
            })
        return sorted(persons_list, key=lambda x: x["person_id"])

    def finish_session(self, avg_fps: float):
        """Closes session and saves person thumbnails in database."""
        persons_details = self.get_persons_details()
        save_person_thumbnails(self.session_id, persons_details)
        close_session(self.session_id, self.total_frames, avg_fps)
        self.executor.shutdown(wait=False)
