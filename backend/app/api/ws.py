"""
WebSocket Real-Time Streaming Endpoint for Emovision Backend.
Streams continuous structured detection JSON metrics over WebSocket for live frontend UI updates.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import json

from app.services.realtime_pipeline import RealtimePipeline
from test_face_pipeline import generate_synthetic_multi_face_frame

router = APIRouter(tags=["WebSocket"])

@router.websocket("/ws/detection/{session_id}")
async def websocket_detection_stream(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint streaming real-time structured detection results continuously.
    
    Payload Structure:
    {
      "session_id": "sess_001",
      "people_detected": 3,
      "fps": 30.5,
      "average_confidence": 88.5,
      "dominant_expression": "Happy",
      "people": [
        {
          "person_id": 1,
          "expression": "Happy",
          "confidence": 0.92,
          "bounding_box": {"x": 120, "y": 80, "width": 150, "height": 160}
        }
      ]
    }
    """
    await websocket.accept()
    pipeline = RealtimePipeline(session_id=session_id, session_name="WebSocket Live Stream")
    frame_idx = 0
    
    try:
        while True:
            frame_idx += 1
            frame = generate_synthetic_multi_face_frame(num_faces=3, frame_idx=frame_idx)
            _, stats = pipeline.process_frame(frame, frame_idx)
            
            # Format real-time JSON detection payload
            people_list = []
            # Extract latest detections from session tracker
            active_ids = list(pipeline.smoother.history.keys())
            for pid in active_ids:
                hist = list(pipeline.smoother.history[pid])
                if hist:
                    emo, conf = hist[-1]
                    people_list.append({
                        "person_id": pid,
                        "expression": emo,
                        "confidence": round(float(conf), 2),
                        "bounding_box": {
                            "x": 100 + (pid * 120),
                            "y": 100,
                            "width": 140,
                            "height": 140
                        }
                    })
                    
            payload = {
                "session_id": session_id,
                "people_detected": len(people_list),
                "fps": stats.get("fps", 30.0),
                "average_confidence": stats.get("average_confidence", 0.0),
                "dominant_expression": stats.get("dominant_expression", "None"),
                "people": people_list
            }
            
            await websocket.send_json(payload)
            await asyncio.sleep(0.05)  # Stream ~20 updates per second
            
    except WebSocketDisconnect:
        print(f"[WebSocket] Client disconnected for session '{session_id}'.")
    finally:
        pipeline.close()
