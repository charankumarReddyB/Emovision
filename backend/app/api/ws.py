"""
WebSocket Real-Time Streaming Endpoint for Emovision Backend.
Processes actual webcam frames using OpenCV Face Detection, Centroid Person Tracking,
and PyTorch Facial Expression Classification.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import base64
import cv2
import numpy as np
import json

from app.services.realtime_pipeline import RealtimePipeline

router = APIRouter(tags=["WebSocket"])

@router.websocket("/ws/detection/{session_id}")
async def websocket_detection_stream(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint receiving webcam frames from frontend client or local camera feed,
    executing real OpenCV face detection + PyTorch facial expression classifier,
    and returning real face bounding box coordinates and emotion metrics.
    """
    await websocket.accept()
    pipeline = RealtimePipeline(session_id=session_id, session_name="WebSocket Live Stream")
    frame_idx = 0
    cap = None
    
    try:
        while True:
            frame = None
            
            # 1. Check for incoming base64 video frame from client
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.04)
                if data:
                    if "," in data:
                        data = data.split(",")[1]
                    img_bytes = base64.b64decode(data)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                pass

            # 2. Fallback to local OpenCV webcam if client doesn't stream base64 frame
            if frame is None or frame.size == 0:
                if cap is None:
                    cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    ret, local_frame = cap.read()
                    if ret and local_frame is not None and local_frame.size > 0:
                        frame = local_frame

            if frame is None or frame.size == 0:
                await asyncio.sleep(0.04)
                continue

            frame_idx += 1
            
            # 3. Execute Real OpenCV Face Detection + Centroid Tracking + PyTorch Classifier
            raw_dets = pipeline.detector.detect_faces(frame)
            tracked_dets = pipeline.tracker.update(raw_dets)
            
            active_ids = [d.get("person_id") for d in tracked_dets if d.get("person_id")]
            pipeline.smoother.cleanup_inactive_tracks(active_ids)
            
            classified_dets = pipeline.classifier.classify_tracked_faces(tracked_dets, frame_idx)
            
            for det in classified_dets:
                pid = det.get("person_id", -1)
                raw_emo = det.get("emotion", "Neutral")
                raw_conf = det.get("emotion_confidence", 0.5)
                if pid > 0:
                    smoothed_emo, smoothed_conf = pipeline.smoother.smooth_prediction(pid, raw_emo, raw_conf)
                    det["emotion"] = smoothed_emo
                    det["emotion_confidence"] = smoothed_conf

            current_fps = pipeline.fps_counter.update()
            live_stats = pipeline.session_tracker.process_frame_detections(frame_idx, classified_dets)
            live_stats["fps"] = current_fps

            # 4. Format Real Bounding Box & Emotion Payload
            people_list = []
            h, w = frame.shape[:2]
            for det in classified_dets:
                pid = det.get("person_id", 1)
                bx, by, bw, bh = det.get("bbox", [0, 0, 0, 0])
                people_list.append({
                    "person_id": pid,
                    "expression": det.get("emotion", "Neutral"),
                    "confidence": round(float(det.get("emotion_confidence", 0.8)), 2),
                    "bounding_box": {
                        "x": int(bx),
                        "y": int(by),
                        "width": int(bw),
                        "height": int(bh),
                        "frame_width": int(w),
                        "frame_height": int(h)
                    }
                })

            payload = {
                "session_id": session_id,
                "people_detected": len(people_list),
                "fps": round(float(current_fps), 1),
                "average_confidence": round(float(live_stats.get("average_confidence", 0.0)), 1),
                "dominant_expression": live_stats.get("dominant_expression", "Neutral"),
                "people": people_list
            }

            await websocket.send_json(payload)
            await asyncio.sleep(0.03)

    except WebSocketDisconnect:
        print(f"[WebSocket] Client disconnected for session '{session_id}'.")
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
        pipeline.close()
