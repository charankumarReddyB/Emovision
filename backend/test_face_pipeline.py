"""
Test Script for Emovision Computer Vision & Emotion Recognition Pipeline.
Verifies N-Face Detection, Face Tracking (Person IDs), Emotion Classification, and FPS Counter.

Usage:
  python backend/test_face_pipeline.py --mode synthetic   # Automated synthetic N-face test
  python backend/test_face_pipeline.py --mode webcam      # Live webcam detection, tracking & emotions
  python backend/test_face_pipeline.py --mode image --path path/to/image.jpg
"""
import cv2
import numpy as np
import argparse
import time
import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from app.services.face_detector import FaceDetector
from app.services.face_tracker import FaceTracker
from app.services.preprocessing import FacePreprocessor
from app.services.fps_counter import FPSCounter
from app.services.emotion_classifier import EmotionClassifier
from app.db.database import init_db, create_session, log_frame_detections, close_session

def draw_face_annotations(frame: np.ndarray, detections: list, fps: float):
    """
    Draws bounding boxes, Person IDs, Emotion Predictions, confidence scores, and FPS on frame.
    """
    annotated = frame.copy()
    
    # Header Overlay
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 42), (20, 20, 20), -1)
    status_str = f"EMOVISION CV PIPELINE | Detected Faces (N): {len(detections)} | FPS: {fps:.1f}"
    cv2.putText(annotated, status_str, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    
    colors = [
        (0, 255, 0), (255, 165, 0), (255, 0, 255), (0, 215, 255),
        (255, 255, 0), (0, 128, 255), (128, 255, 0), (255, 100, 100)
    ]
    
    for det in detections:
        x, y, w, h = det["bbox"]
        pid = det.get("person_id", -1)
        conf = det.get("confidence", 0.0)
        emotion = det.get("emotion", "Neutral")
        emo_conf = det.get("emotion_confidence", 0.0)
        
        color = colors[(pid - 1) % len(colors)] if pid > 0 else (0, 255, 0)
        
        # Bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        
        # Label box (Person ID -> Emotion -> Conf)
        label_text = f"Person {pid}: {emotion} ({emo_conf*100:.0f}%)" if pid > 0 else f"{emotion} ({emo_conf*100:.0f}%)"
        (lbl_w, lbl_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        
        cv2.rectangle(annotated, (x, max(0, y - lbl_h - 10)), (x + lbl_w + 6, max(0, y)), color, -1)
        cv2.putText(
            annotated,
            label_text,
            (x + 3, max(12, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
        
    return annotated

def generate_synthetic_multi_face_frame(num_faces: int, frame_idx: int, width: int = 800, height: int = 600) -> np.ndarray:
    """
    Generates synthetic frames containing N moving faces to test N-face detection & tracking.
    """
    frame = np.ones((height, width, 3), dtype=np.uint8) * 230
    
    for x in range(0, width, 40):
        cv2.line(frame, (x, 0), (x, height), (210, 210, 210), 1)
    for y in range(0, height, 40):
        cv2.line(frame, (0, y), (width, y), (210, 210, 210), 1)
        
    for i in range(num_faces):
        speed = (i + 1) * 3
        offset_x = (frame_idx * speed + i * 180) % (width - 150) + 50
        offset_y = 120 + (i % 3) * 140 + int(np.sin(frame_idx * 0.1 + i) * 15)
        
        center_x, center_y = int(offset_x), int(offset_y)
        face_radius = 45
        
        cv2.circle(frame, (center_x, center_y), face_radius, (180, 200, 230), -1)
        cv2.circle(frame, (center_x, center_y), face_radius, (50, 50, 50), 2)
        cv2.circle(frame, (center_x - 15, center_y - 12), 6, (40, 40, 40), -1)
        cv2.circle(frame, (center_x + 15, center_y - 12), 6, (40, 40, 40), -1)
        cv2.ellipse(frame, (center_x, center_y + 15), (15, 8), 0, 0, 180, (40, 40, 40), 2)
        
    return frame

def run_synthetic_test(num_faces: int = 3, num_frames: int = 50):
    """
    Executes automated synthetic N-face detection, tracking, & emotion classification test sequence.
    """
    print(f"\n=======================================================")
    print(f"RUNNING AUTOMATED SYNTHETIC MULTI-FACE & EMOTION TEST ({num_faces} Faces, {num_frames} Frames)")
    print(f"=======================================================")
    
    init_db()
    session_id = "test_synth_emotion_001"
    create_session(session_id, "Synthetic Emotion Test Session", "synthetic")
    
    detector = FaceDetector()
    tracker = FaceTracker()
    emotion_classifier = EmotionClassifier()
    fps_counter = FPSCounter()
    fps_counter.start()
    
    detected_counts = []
    unique_person_ids = set()
    sample_predictions = []
    
    for frame_idx in range(num_frames):
        frame = generate_synthetic_multi_face_frame(num_faces, frame_idx)
        
        # 1. Detect faces
        raw_detections = detector.detect_faces(frame)
        
        # 2. Track faces across frames
        tracked_detections = tracker.update(raw_detections)
        
        # 3. Classify facial expressions per face
        classified_detections = emotion_classifier.classify_tracked_faces(tracked_detections, frame_idx)
        
        # 4. Update FPS
        current_fps = fps_counter.update()
        
        # 5. Log to SQLite
        log_frame_detections(session_id, frame_idx, classified_detections)
        
        for det in classified_detections:
            if det.get("person_id"):
                unique_person_ids.add(det["person_id"])
                
        detected_counts.append(len(classified_detections))
        
        if frame_idx % 10 == 0 or frame_idx == num_frames - 1:
            preds_str = [f"Person {d.get('person_id')}: {d.get('emotion')} ({d.get('emotion_confidence')*100:.0f}%)" for d in classified_detections]
            sample_predictions = preds_str
            print(f"Frame {frame_idx:02d}: Detected {len(classified_detections)} faces | Predictions: {preds_str} | FPS: {current_fps:.1f}")

    close_session(session_id, num_frames, fps_counter.get_avg_fps())
    
    print("\n---------------- TEST RESULTS SUMMARY ----------------")
    print(f"Total Frames Processed: {num_frames}")
    print(f"Average FPS: {fps_counter.get_avg_fps():.1f}")
    print(f"Max Faces Detected in single frame (N): {max(detected_counts) if detected_counts else 0}")
    print(f"Unique Person IDs tracked across session: {sorted(list(unique_person_ids))}")
    print(f"Sample Real-time Predictions: {sample_predictions}")
    print(f"Session results saved to SQLite database.")
    print("------------------------------------------------------\n")
    return True

def run_webcam_test():
    """
    Runs live webcam multi-face detection, tracking, and emotion recognition.
    """
    print("\nLaunching Live Webcam Test... Press 'q' or ESC to exit.")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not access webcam (Camera ID 0). Check camera connection.")
        return
        
    detector = FaceDetector()
    tracker = FaceTracker()
    emotion_classifier = EmotionClassifier()
    fps_counter = FPSCounter()
    fps_counter.start()
    
    frame_idx = 0
    init_db()
    session_id = f"webcam_emotion_{int(time.time())}"
    create_session(session_id, "Live Webcam Emotion Session", "webcam")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            raw_detections = detector.detect_faces(frame)
            tracked_detections = tracker.update(raw_detections)
            classified_detections = emotion_classifier.classify_tracked_faces(tracked_detections, frame_idx)
            current_fps = fps_counter.update()
            
            log_frame_detections(session_id, frame_idx, classified_detections)
            annotated = draw_face_annotations(frame, classified_detections, current_fps)
            cv2.imshow("Emovision - Real-Time Multi-Face Emotion Recognition", annotated)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        close_session(session_id, frame_idx, fps_counter.get_avg_fps())
        print(f"Webcam session finished. Processed {frame_idx} frames at {fps_counter.get_avg_fps()} avg FPS.")

def main():
    parser = argparse.ArgumentParser(description="Emovision Real-time Emotion Recognition Pipeline")
    parser.add_argument("--mode", type=str, default="synthetic", choices=["synthetic", "webcam", "image"],
                        help="Test mode: 'synthetic' (default), 'webcam', or 'image'")
    parser.add_argument("--path", type=str, default="", help="Path to image file if mode is image")
    
    args = parser.parse_args()
    
    if args.mode == "synthetic":
        run_synthetic_test()
    elif args.mode == "webcam":
        run_webcam_test()
    elif args.mode == "image":
        if not args.path or not Path(args.path).exists():
            print(f"ERROR: Image file path '{args.path}' not found.")
            return
        frame = cv2.imread(args.path)
        detector = FaceDetector()
        tracker = FaceTracker()
        emotion_classifier = EmotionClassifier()
        
        detections = detector.detect_faces(frame)
        tracked = tracker.update(detections)
        classified = emotion_classifier.classify_tracked_faces(tracked, 0)
        annotated = draw_face_annotations(frame, classified, 0.0)
        
        out_path = "output_emotion_detected.jpg"
        cv2.imwrite(out_path, annotated)
        print(f"Processed image. Detected {len(classified)} faces. Saved output to {out_path}")

if __name__ == "__main__":
    main()
