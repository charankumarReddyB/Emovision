"""
Master Real-Time Multi-Face Facial Expression Recognition Pipeline.
Integrates SCRFD Face Detection, 5-Point Affine Geometric Face Alignment,
MobileFaceNet ONNX Expression Classification, Live Statistics, and HUD Rendering.
"""
import cv2
import numpy as np
import time
from typing import Tuple, Dict, Any, List, Optional

from app.services.face_detector import FaceDetector
from app.services.emotion_classifier import EmotionClassifier
from app.services.session_tracker import SessionTracker
from app.services.fps_counter import FPSCounter

class RealtimePipeline:
    """
    End-to-End Real-Time Multi-Face Facial Expression Recognition Engine.
    Detects faces independently per frame without persistent person tracking.
    """
    def __init__(self, session_id: str = "live_session", session_name: str = "Real-Time Pipeline"):
        self.detector = FaceDetector()
        self.classifier = EmotionClassifier()
        self.session_tracker = SessionTracker(session_id, session_name=session_name)
        self.fps_counter = FPSCounter()
        self.fps_counter.start()
        
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        annotated_frame, live_stats, _ = self.process_frame_with_detections(frame, frame_idx)
        return annotated_frame, live_stats

    def process_frame_with_detections(
        self,
        frame: np.ndarray,
        frame_idx: int
    ) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Executes SCRFD face detection + 5-point keypoint alignment + batch MobileFaceNet ONNX classification.
        
        Args:
            frame (np.ndarray): BGR OpenCV image frame.
            frame_idx (int): Current frame sequence index.
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
                (Annotated Frame, Live Statistics Dict, Classified Detections List)
        """
        if frame is None or frame.size == 0:
            return frame, {}, []

        # 1. Detect human faces in frame with 5 facial keypoints using SCRFD
        raw_detections = self.detector.detect_faces(frame)
        
        # 2. Extract 5-point aligned face chips for batch inference
        aligned_chips = [det["aligned_chip"] for det in raw_detections]
        
        # 3. Batch emotion classification for N faces
        if aligned_chips:
            bboxes = [det["bbox"] for det in raw_detections]
            emotions_list = self.classifier.classify_batch(aligned_chips, bboxes=bboxes)
        else:
            emotions_list = []
            
        # 4. Format classified face detections
        classified_detections = []
        for idx, (det, (emotion, conf)) in enumerate(zip(raw_detections, emotions_list), start=1):
            classified_detections.append({
                "person_id": idx,
                "face_index": idx,
                "bbox": det["bbox"],
                "kps": det.get("kps"),
                "detection_confidence": det["confidence"],
                "emotion": emotion,
                "emotion_confidence": conf,
                "aligned_chip": det.get("aligned_chip"),
                "face_chip": det.get("face_chip")
            })

        # 5. Update FPS
        current_fps = self.fps_counter.update()
        
        # 6. Update Session Tracker Statistics
        live_stats = self.session_tracker.process_frame_detections(frame_idx, classified_detections)
        live_stats["fps"] = current_fps
        
        # 7. Render Visual HUD Overlay
        annotated_frame = self.render_hud(frame, classified_detections, live_stats)
        
        return annotated_frame, live_stats, classified_detections

    def render_hud(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        stats: Dict[str, Any]
    ) -> np.ndarray:
        """
        Renders face bounding boxes, Face # labels, expressions, confidence %, and live HUD statistics.
        """
        hud = frame.copy()
        height, width = hud.shape[:2]
        
        # Colors palette
        colors = [
            (0, 215, 255), (0, 255, 0), (255, 0, 255), (255, 165, 0),
            (255, 255, 0), (0, 128, 255), (128, 255, 0), (255, 100, 100)
        ]

        # Draw Face Bounding Boxes & Labels
        for det in detections:
            x, y, w, h = det["bbox"]
            f_idx = det.get("face_index", 1)
            emotion = det.get("emotion", "Neutral")
            conf = det.get("emotion_confidence", 0.0)
            
            color = colors[(f_idx - 1) % len(colors)]
            
            # Draw Box
            cv2.rectangle(hud, (x, y), (x + w, y + h), color, 2)
            
            # Draw 5 Keypoints if present
            kps = det.get("kps")
            if kps is not None:
                for kp in kps:
                    kx, ky = int(kp[0]), int(kp[1])
                    cv2.circle(hud, (kx, ky), 2, (0, 255, 255), -1)
            
            # Label string: "Face 1 -- Happy -- 92%"
            label_str = f"Face {f_idx} -- {emotion} -- {conf*100:.0f}%"
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
        header_text = f"EMOVISION LIVE | Visible Faces (N): {stats.get('total_people', len(detections))} | FPS: {stats.get('fps', 0.0):.1f}"
        cv2.putText(hud, header_text, (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        # Right Side Live Statistics Overlay Panel
        panel_w = 260
        panel_h = 150
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

        return hud

    def close(self):
        """Finalizes session."""
        self.session_tracker.finish_session(self.fps_counter.get_avg_fps())
