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
import logging

from app.services.realtime_pipeline import RealtimePipeline

logger = logging.getLogger("emovision.ws")
logging.basicConfig(level=logging.INFO)

router = APIRouter(tags=["WebSocket"])

@router.websocket("/ws/detection/{session_id}")
async def websocket_detection_stream(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint streaming real-time human face detections and facial expression metrics.
    Operates in non-blocking async mode for 30+ FPS zero-latency performance.
    """
    client_host = getattr(websocket.client, "host", "unknown") if websocket.client else "unknown"
    client_port = getattr(websocket.client, "port", "unknown") if websocket.client else "unknown"
    logger.info(f"[WebSocket] Connection attempt received. Session ID: {session_id} | Client: {client_host}:{client_port}")

    await websocket.accept()
    logger.info(f"[WebSocket] Connection ACCEPTED for Session ID: {session_id}")

    pipeline = RealtimePipeline(session_id=session_id, session_name="WebSocket Live Stream")
    frame_idx = 0
    
    try:
        while True:
            # 1. Receive latest base64 video frame from client
            try:
                data = await websocket.receive_text()
            except WebSocketDisconnect as e:
                logger.info(f"[WebSocket] Client disconnected cleanly via WebSocketDisconnect. Session ID: {session_id} | Code: {getattr(e, 'code', 'N/A')}")
                break
            except Exception as e:
                logger.warning(f"[WebSocket] Frame receive exception. Session ID: {session_id} | Reason: {str(e)}")
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
            except Exception as e:
                logger.warning(f"[WebSocket] Frame base64 decode error in Session '{session_id}': {e}")
                continue

            if frame is None or frame.size == 0:
                await asyncio.sleep(0.005)
                continue

            frame_idx += 1
            if frame_idx % 100 == 1:
                logger.info(f"[WebSocket] Session ID '{session_id}' processing frame #{frame_idx}")
            
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
            except Exception as e:
                logger.warning(f"[WebSocket] send_text exception in Session '{session_id}': {e}")
                break
                
            await asyncio.sleep(0.002)

    except WebSocketDisconnect as e:
        logger.info(f"[WebSocket] Client disconnected. Session ID: {session_id} | Code: {getattr(e, 'code', 'N/A')} | Reason: {getattr(e, 'reason', 'N/A')}")
    except Exception as e:
        logger.error(f"[WebSocket] Stream exception for Session ID '{session_id}': {e}", exc_info=True)
    finally:
        logger.info(f"[WebSocket] Closing pipeline for Session ID: {session_id} (Total frames processed: {frame_idx})")
        pipeline.close()
