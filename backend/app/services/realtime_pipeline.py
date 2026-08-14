"""
Master Real-Time Multi-Person Facial Expression Recognition Pipeline.
Integrates Face Detection, Person Tracking, Emotion Classification, Temporal Prediction Smoothing,
Live Statistics, and HUD Overlay Rendering.
"""
import cv2
import numpy as np
import time
from typing import Tuple, Dict, Any, List, Optional

from app.services.face_detector import FaceDetector
from app.services.face_tracker import FaceTracker
from app.services.emotion_classifier import EmotionClassifier
from app.services.prediction_smoother import PredictionSmoother
from app.services.session_tracker import SessionTracker
from app.services.fps_counter import FPSCounter

class RealtimePipeline:
    """
    End-to-End Real-Time Multi-Person Facial Expression Recognition Engine.
    """
    def __init__(self, session_id: str = "live_session", session_name: str = "Real-Time Pipeline"):
        self.detector = FaceDetector()
        self.tracker = FaceTracker()
        self.classifier = EmotionClassifier()
        self.smoother = PredictionSmoother(window_size=5)
        self.session_tracker = SessionTracker(session_id, session_name=session_name)
        self.fps_counter = FPSCounter()
        self.fps_counter.start()
        
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes real-time processing pipeline on input video/webcam frame.
        
        Args:
            frame (np.ndarray): BGR OpenCV image frame.
            frame_idx (int): Current frame sequence index.
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (Annotated OpenCV Frame, Live Statistics Dict)
        """
        if frame is None or frame.size == 0:
            return frame, {}

        # 1. Detect N faces
        raw_detections = self.detector.detect_faces(frame)
        
        # 2. Track faces across frames (Assign & Maintain Person IDs)
        tracked_detections = self.tracker.update(raw_detections)
        
        # 3. Cleanup inactive smoother tracks
        active_ids = [d.get("person_id") for d in tracked_detections if d.get("person_id")]
        self.smoother.cleanup_inactive_tracks(active_ids)
        
        # 4. Classify facial expressions
        classified_detections = self.classifier.classify_tracked_faces(tracked_detections, frame_idx)
        
        # 5. Apply Temporal Prediction Smoothing per Person ID
        for det in classified_detections:
            pid = det.get("person_id", -1)
            raw_emo = det.get("emotion", "Neutral")
            raw_conf = det.get("emotion_confidence", 0.5)
            
            if pid > 0:
                smoothed_emo, smoothed_conf = self.smoother.smooth_prediction(pid, raw_emo, raw_conf)
                det["emotion"] = smoothed_emo
                det["emotion_confidence"] = smoothed_conf

        # 6. Update FPS
        current_fps = self.fps_counter.update()
        
        # 7. Update Session Tracker Statistics
        live_stats = self.session_tracker.process_frame_detections(frame_idx, classified_detections)
        live_stats["fps"] = current_fps
        
        # 8. Render Visual HUD Overlay
        annotated_frame = self.render_hud(frame, classified_detections, live_stats)
        
        return annotated_frame, live_stats

    def render_hud(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        stats: Dict[str, Any]
    ) -> np.ndarray:
        """
        Renders bounding boxes, Person IDs, smoothed emotions, confidence %, and live HUD statistics.
        """
        hud = frame.copy()
        height, width = hud.shape[:2]
        
        # Colors palette
        colors = [
            (0, 255, 0), (255, 165, 0), (255, 0, 255), (0, 215, 255),
            (255, 255, 0), (0, 128, 255), (128, 255, 0), (255, 100, 100)
        ]

        # Draw Face Bounding Boxes & Labels
        for det in detections:
            x, y, w, h = det["bbox"]
            pid = det.get("person_id", -1)
            emotion = det.get("emotion", "Neutral")
            conf = det.get("emotion_confidence", 0.0)
            
            color = colors[(pid - 1) % len(colors)] if pid > 0 else (0, 255, 0)
            
            # Draw Box
            cv2.rectangle(hud, (x, y), (x + w, y + h), color, 2)
            
            # Label string: "Person 1 — Happy — 92%"
            label_str = f"Person {pid} -- {emotion} -- {conf*100:.0f}%" if pid > 0 else f"{emotion} -- {conf*100:.0f}%"
            (lbl_w, lbl_h), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            
            # Label background box
            cv2.rectangle(hud, (x, max(0, y - lbl_h - 10)), (x + lbl_w + 8, max(0, y)), color, -1)
            cv2.putText(
                hud,
                label_str,
                (x + 4, max(12, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        # Top Header Bar
        cv2.rectangle(hud, (0, 0), (width, 42), (25, 25, 25), -1)
        header_text = f"EMOVISION LIVE | Visible People (N): {stats.get('total_people', 0)} | FPS: {stats.get('fps', 0.0):.1f}"
        cv2.putText(hud, header_text, (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        # Right Side Live Statistics Dashboard Overlay Panel
        panel_w = 260
        panel_h = 160
        cv2.rectangle(hud, (width - panel_w - 10, 50), (width - 10, 50 + panel_h), (20, 20, 20), -1)
        cv2.rectangle(hud, (width - panel_w - 10, 50), (width - 10, 50 + panel_h), (0, 255, 255), 1)

        y_offset = 72
        cv2.putText(hud, "LIVE SESSION STATISTICS", (width - panel_w, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)
        
        y_offset += 22
        cv2.putText(hud, f"Dominant: {stats.get('dominant_expression', 'None')}", (width - panel_w, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.putText(hud, f"Avg Confidence: {stats.get('average_confidence', 0.0):.1f}%", (width - panel_w, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.putText(hud, f"Duration: {stats.get('session_duration_sec', 0.0):.1f}s", (width - panel_w, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        y_offset += 20
        breakdown_str = "Counts: " + ", ".join([f"{k}:{v}" for k, v in stats.get('expression_counts', {}).items()])
        if len(breakdown_str) > 28:
            breakdown_str = breakdown_str[:28] + "..."
        cv2.putText(hud, breakdown_str, (width - panel_w, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        return hud

    def close(self):
        """Finalizes tracking session."""
        self.session_tracker.finish_session(self.fps_counter.get_avg_fps())
