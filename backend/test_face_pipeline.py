"""
Test Script for Emovision Computer Vision Foundation.
Verifies N-Face Detection, Face Tracking (Person IDs), Preprocessing, and FPS Counter.

Usage:
  python test_face_pipeline.py --mode synthetic   # Automated synthetic N-face test
  python test_face_pipeline.py --mode webcam      # Live webcam detection & tracking
  python test_face_pipeline.py --mode image --path path/to/image.jpg
  python test_face_pipeline.py --mode video --path path/to/video.mp4
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
from app.db.database import init_db, create_session, log_frame_detections, close_session

def draw_face_annotations(frame: np.ndarray, detections: list, fps: float):
    """
    Draws bounding boxes, Person IDs, confidence scores, and system FPS on the image frame.
    """
    annotated = frame.copy()
    
    # Header Overlay
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 40), (20, 20, 20), -1)
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
        
        color = colors[(pid - 1) % len(colors)] if pid > 0 else (0, 255, 0)
        
        # Bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        
        # Label box
        label_text = f"Person {pid} ({conf*100:.0f}%)" if pid > 0 else f"Face ({conf*100:.0f}%)"
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
    Generates a realistic synthetic frame containing N moving faces to test N-face detection & tracking.
    """
    frame = np.ones((height, width, 3), dtype=np.uint8) * 230  # Light background
    
    # Draw simple background grid
    for x in range(0, width, 40):
        cv2.line(frame, (x, 0), (x, height), (210, 210, 210), 1)
    for y in range(0, height, 40):
        cv2.line(frame, (0, y), (width, y), (210, 210, 210), 1)
        
    # Draw N synthetic face shapes moving horizontally across frame
    for i in range(num_faces):
        # Calculate moving center position for face i
        speed = (i + 1) * 3
        offset_x = (frame_idx * speed + i * 180) % (width - 150) + 50
        offset_y = 120 + (i % 3) * 140 + int(np.sin(frame_idx * 0.1 + i) * 15)
        
        center_x, center_y = int(offset_x), int(offset_y)
        face_radius = 45
        
        # Face skin tone circle
        cv2.circle(frame, (center_x, center_y), face_radius, (180, 200, 230), -1)
        cv2.circle(frame, (center_x, center_y), face_radius, (50, 50, 50), 2)
        
        # Eyes
        cv2.circle(frame, (center_x - 15, center_y - 12), 6, (40, 40, 40), -1)
        cv2.circle(frame, (center_x + 15, center_y - 12), 6, (40, 40, 40), -1)
        
        # Mouth
        cv2.ellipse(frame, (center_x, center_y + 15), (15, 8), 0, 0, 180, (40, 40, 40), 2)
        
    return frame

def run_synthetic_test(num_faces: int = 3, num_frames: int = 50):
    """
    Executes an automated synthetic N-face detection & tracking test sequence.
    """
    print(f"\n=======================================================")
    print(f"RUNNING AUTOMATED SYNTHETIC MULTI-FACE TEST ({num_faces} Faces, {num_frames} Frames)")
    print(f"=======================================================")
    
    init_db()
    session_id = "test_synth_001"
    create_session(session_id, "Synthetic Test Session", "synthetic")
    
    detector = FaceDetector()
    tracker = FaceTracker()
    preprocessor = FacePreprocessor()
    fps_counter = FPSCounter()
    fps_counter.start()
    
    detected_counts = []
    unique_person_ids = set()
    
    for frame_idx in range(num_frames):
        frame = generate_synthetic_multi_face_frame(num_faces, frame_idx)
        
        # 1. Detect faces
        raw_detections = detector.detect_faces(frame)
        
        # 2. Track faces across frames & assign Person IDs
        tracked_detections = tracker.update(raw_detections)
        
        # 3. Update FPS
        current_fps = fps_counter.update()
        
        # 4. Log detections to SQLite
        log_frame_detections(session_id, frame_idx, tracked_detections)
        
        # 5. Preprocess each cropped face chip
        for det in tracked_detections:
            face_tensor = preprocessor.preprocess(det["face_chip"])
            det["tensor_shape"] = face_tensor.shape
            if det.get("person_id"):
                unique_person_ids.add(det["person_id"])
                
        detected_counts.append(len(tracked_detections))
        
        if frame_idx % 10 == 0 or frame_idx == num_frames - 1:
            pids = [d.get("person_id") for d in tracked_detections]
            print(f"Frame {frame_idx:02d}: Detected {len(tracked_detections)} faces | Assigned Person IDs: {pids} | FPS: {current_fps:.1f}")

    close_session(session_id, num_frames, fps_counter.get_avg_fps())
    
    print("\n---------------- TEST RESULTS SUMMARY ----------------")
    print(f"Total Frames Processed: {num_frames}")
    print(f"Average FPS: {fps_counter.get_avg_fps():.1f}")
    print(f"Max Faces Detected in single frame (N): {max(detected_counts) if detected_counts else 0}")
    print(f"Unique Person IDs tracked across session: {sorted(list(unique_person_ids))}")
    print(f"Session results saved to SQLite database.")
    print("------------------------------------------------------\n")
    return True

def run_webcam_test():
    """
    Runs live webcam detection & tracking.
    """
    print("\nLaunching Live Webcam Test... Press 'q' or ESC to exit.")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not access webcam (Camera ID 0). Please check camera permissions or connection.")
        return
        
    detector = FaceDetector()
    tracker = FaceTracker()
    preprocessor = FacePreprocessor()
    fps_counter = FPSCounter()
    fps_counter.start()
    
    frame_idx = 0
    init_db()
    session_id = f"webcam_{int(time.time())}"
    create_session(session_id, "Live Webcam Session", "webcam")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera.")
                break
                
            frame_idx += 1
            raw_detections = detector.detect_faces(frame)
            tracked_detections = tracker.update(raw_detections)
            current_fps = fps_counter.update()
            
            # Log to SQLite
            log_frame_detections(session_id, frame_idx, tracked_detections)
            
            # Draw overlay
            annotated = draw_face_annotations(frame, tracked_detections, current_fps)
            cv2.imshow("Emovision - Real-Time Multi-Face Detection & Tracking", annotated)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        close_session(session_id, frame_idx, fps_counter.get_avg_fps())
        print(f"Webcam session finished. Processed {frame_idx} frames at {fps_counter.get_avg_fps()} avg FPS.")

def main():
    parser = argparse.ArgumentParser(description="Emovision CV Foundation Test Script")
    parser.add_argument("--mode", type=str, default="synthetic", choices=["synthetic", "webcam", "image", "video"],
                        help="Test mode: 'synthetic' (default), 'webcam', 'image', or 'video'")
    parser.add_argument("--path", type=str, default="", help="Path to image or video file if mode is image/video")
    
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
        preprocessor = FacePreprocessor()
        
        detections = detector.detect_faces(frame)
        tracked = tracker.update(detections)
        annotated = draw_face_annotations(frame, tracked, 0.0)
        
        out_path = "output_detected.jpg"
        cv2.imwrite(out_path, annotated)
        print(f"Processed image. Detected {len(tracked)} faces. Saved output to {out_path}")

if __name__ == "__main__":
    main()
