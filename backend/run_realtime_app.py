"""
Standalone Real-Time Multi-Person Facial Expression Recognition Application.
Integrates Webcam / Video Feed / Synthetic Stream with Face Detection, Multi-Person Tracking,
Temporal Prediction Smoothing, Live Statistics HUD, and SQLite Session Logging.

Usage:
  python backend/run_realtime_app.py --mode synthetic   # Automated multi-person simulation
  python backend/run_realtime_app.py --mode webcam      # Live webcam stream
  python backend/run_realtime_app.py --mode video --path video.mp4
"""
import cv2
import argparse
import time
import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from app.services.realtime_pipeline import RealtimePipeline
from test_face_pipeline import generate_synthetic_multi_face_frame

def run_synthetic_simulation(num_faces: int = 3, num_frames: int = 60):
    """Runs automated multi-person synthetic simulation."""
    print("\n=======================================================")
    print(f"LAUNCHING EMOVISION REAL-TIME SIMULATION ({num_faces} Faces, {num_frames} Frames)")
    print("=======================================================")
    
    session_id = f"sim_{int(time.time())}"
    pipeline = RealtimePipeline(session_id=session_id, session_name="Synthetic Real-Time Simulation")
    
    for frame_idx in range(num_frames):
        # Vary face count dynamically to simulate entering/leaving frame
        current_num_faces = num_faces if frame_idx < 40 else num_faces + 1
        frame = generate_synthetic_multi_face_frame(current_num_faces, frame_idx)
        
        annotated_frame, stats = pipeline.process_frame(frame, frame_idx)
        
        if frame_idx % 15 == 0 or frame_idx == num_frames - 1:
            print(f"Frame [{frame_idx:02d}/{num_frames:02d}] | "
                  f"Visible People (N): {stats.get('total_people', 0)} | "
                  f"Dominant: {stats.get('dominant_expression')} | "
                  f"Avg Conf: {stats.get('average_confidence')} | "
                  f"FPS: {stats.get('fps'):.1f}")
                  
    pipeline.close()
    print("\n---------------- SIMULATION SUMMARY RESULTS ----------------")
    print(f"Total Frames Processed: {num_frames}")
    print(f"Session Duration: {stats.get('session_duration_sec')}s")
    print(f"Average FPS: {pipeline.fps_counter.get_avg_fps():.1f}")
    print(f"Session data saved to SQLite database.")
    print("-----------------------------------------------------------\n")

def run_webcam_app():
    """Runs live webcam application."""
    print("\nLaunching Live Webcam Application... Press 'q' or ESC to exit.")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not access webcam (Camera ID 0). Check connection.")
        return
        
    session_id = f"webcam_{int(time.time())}"
    pipeline = RealtimePipeline(session_id=session_id, session_name="Live Webcam Session")
    
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            annotated_frame, stats = pipeline.process_frame(frame, frame_idx)
            
            cv2.imshow("Emovision - Real-Time Multi-Person Facial Expression Recognition", annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pipeline.close()
        print(f"Webcam session ended. Processed {frame_idx} frames at {pipeline.fps_counter.get_avg_fps():.1f} avg FPS.")

def main():
    parser = argparse.ArgumentParser(description="Emovision Standalone Real-Time Application")
    parser.add_argument("--mode", type=str, default="synthetic", choices=["synthetic", "webcam", "video"],
                        help="Application mode: 'synthetic' (default), 'webcam', or 'video'")
    parser.add_argument("--path", type=str, default="", help="Video file path if mode is video")
    
    args = parser.parse_args()
    
    if args.mode == "synthetic":
        run_synthetic_simulation()
    elif args.mode == "webcam":
        run_webcam_app()
    elif args.mode == "video":
        if not args.path or not Path(args.path).exists():
            print(f"ERROR: Video file '{args.path}' not found.")
            return
        cap = cv2.VideoCapture(args.path)
        session_id = f"video_{int(time.time())}"
        pipeline = RealtimePipeline(session_id=session_id, session_name="Video File Session")
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            annotated_frame, stats = pipeline.process_frame(frame, frame_idx)
            cv2.imshow("Emovision Video Player", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        pipeline.close()

if __name__ == "__main__":
    main()
