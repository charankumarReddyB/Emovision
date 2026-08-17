"""
WebSocket Real-Time Streaming Endpoint for Emovision Backend.
Processes base64 webcam video frames using YuNet/SCRFD ONNX face detection,
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
    Operates in non-blocking async mode for 30+ FPS zero-latency performance.
    """
    await websocket.accept()
    pipeline = RealtimePipeline(session_id=session_id, session_name="WebSocket Live Stream")
    frame_idx = 0
    
    try:
        while True:
            # 1. Receive latest base64 video frame from client
            try:
                data = await websocket.receive_text()
            except WebSocketDisconnect:
                break
            except Exception:
                break

            if not data:
                await asyncio.sleep(0.005)
                continue

            if "," in data:
                data = data.split(",")[1]

            try:
                img_bytes = base64.b64decode(data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception:
                continue

            if frame is None or frame.size == 0:
                await asyncio.sleep(0.005)
                continue

            frame_idx += 1
            
            # 2. Execute face detection + alignment + MobileFaceNet ONNX classifier
            annotated_frame, live_stats, classified_detections = pipeline.process_frame_with_detections(frame, frame_idx)
            
            h, w = frame.shape[:2]
            people_list = []
            faces_list = []
            
            for det in classified_detections:
                bx, by, bw, bh = det["bbox"]
                emo = str(det["emotion"])
                conf = float(det["emotion_confidence"])
                f_idx = int(det["face_index"])

                faces_list.append({
                    "bbox": [int(bx), int(by), int(bw), int(bh)],
                    "expression": emo,
                    "confidence": round(conf, 4)
                })

                people_list.append({
                    "face_index": f_idx,
                    "person_id": f_idx,
                    "expression": emo,
                    "confidence": round(conf, 4),
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
                "face_count": len(faces_list),
                "people_detected": len(faces_list),
                "fps": round(float(live_stats.get("fps", 30.0)), 1),
                "average_confidence": round(float(live_stats.get("average_confidence", 0.0)), 1),
                "dominant_expression": str(live_stats.get("dominant_expression", "No face detected" if len(faces_list) == 0 else "Neutral")),
                "faces": faces_list,
                "people": people_list
            }

            try:
                await websocket.send_text(json.dumps(payload))
            except Exception:
                break
                
            await asyncio.sleep(0.002)

    except WebSocketDisconnect:
        print(f"[WebSocket] Client disconnected for session '{session_id}'.")
    except Exception as e:
        print(f"[WebSocket] Error in session '{session_id}': {e}")
    finally:
        pipeline.close()
