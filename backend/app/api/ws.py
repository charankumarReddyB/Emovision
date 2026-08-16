"""
WebSocket Real-Time Streaming Endpoint for Emovision Backend.
Processes webcam video frames using OpenCV YuNet DNN face detection
and PyTorch facial expression classifier without person tracking.
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
                await asyncio.sleep(0.05)
                continue

            frame_idx += 1
            
            # 2. Execute Real OpenCV YuNet Face Detection + PyTorch Classifier
            annotated_frame, live_stats = pipeline.process_frame(frame, frame_idx)
            
            # Extract detected faces
            raw_dets = pipeline.detector.detect_faces(frame)
            people_list = []
            h, w = frame.shape[:2]
            
            for idx, det in enumerate(raw_dets, start=1):
                bx, by, bw, bh = det.get("bbox", [0, 0, 0, 0])
                chip = det.get("face_chip")
                if chip is not None and chip.size > 0:
                    emo, conf = pipeline.classifier.classify_face(chip)
                else:
                    emo, conf = "Neutral", 0.50

                people_list.append({
                    "face_index": idx,
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
            await asyncio.sleep(0.03)

    except WebSocketDisconnect:
        print(f"[WebSocket] Client disconnected for session '{session_id}'.")
    finally:
        pipeline.close()
