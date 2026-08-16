"""
WebSocket Real-Time Streaming Endpoint for Emovision Backend.
Processes base64 webcam video frames using SCRFD ONNX face detection,
5-point geometric face alignment, and MobileFaceNet FER ONNX emotion classifier.
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
    WebSocket endpoint streaming real-time human face detections and facial expression metrics.
    Treats each face in frame independently without persistent person IDs.
    """
    await websocket.accept()
    pipeline = RealtimePipeline(session_id=session_id, session_name="WebSocket Live Stream")
    frame_idx = 0
    
    try:
        while True:
            frame = None
            
            # 1. Receive base64 video frame from client
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
                if data:
                    if "," in data:
                        data = data.split(",")[1]
                    img_bytes = base64.b64decode(data)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except asyncio.TimeoutError:
                pass
            except Exception:
                pass

            if frame is None or frame.size == 0:
                await asyncio.sleep(0.02)
                continue

            frame_idx += 1
            
            # 2. Execute SCRFD 5-point face detection + 112x112 face alignment + MobileFaceNet ONNX classifier
            annotated_frame, live_stats, classified_detections = pipeline.process_frame_with_detections(frame, frame_idx)
            
            h, w = frame.shape[:2]
            people_list = []
            
            for det in classified_detections:
                bx, by, bw, bh = det["bbox"]
                emo = det["emotion"]
                conf = det["emotion_confidence"]

                people_list.append({
                    "face_index": det["face_index"],
                    "person_id": det["face_index"],
                    "expression": emo,
                    "confidence": round(float(conf), 2),
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
                "fps": round(float(live_stats.get("fps", 30.0)), 1),
                "average_confidence": round(float(live_stats.get("average_confidence", 0.0)), 1),
                "dominant_expression": live_stats.get("dominant_expression", "No face detected" if len(people_list) == 0 else "Neutral"),
                "people": people_list
            }

            await websocket.send_json(payload)
            await asyncio.sleep(0.02)

    except WebSocketDisconnect:
        print(f"[WebSocket] Client disconnected for session '{session_id}'.")
    finally:
        pipeline.close()
