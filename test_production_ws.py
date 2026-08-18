import asyncio
import json
import base64
import cv2
import numpy as np
import urllib.request
import ssl

async def test_production_websocket():
    # 1. Create a session on production REST API
    api_url = "https://emovision.onrender.com/api/session/start"
    req_data = json.dumps({"session_name": "WS Test Session", "source_type": "webcam"}).encode('utf-8')
    req = urllib.request.Request(api_url, data=req_data, headers={'Content-Type': 'application/json'})
    
    context = ssl._create_unverified_context()
    print("[TEST] 1. Requesting session creation from Render REST API...")
    try:
        with urllib.request.urlopen(req, context=context, timeout=10) as response:
            res_body = json.loads(response.read().decode('utf-8'))
            session_id = res_body["session_id"]
            print(f"[TEST] Session created successfully. Session ID: {session_id}")
    except Exception as e:
        print(f"[TEST ERROR] Failed to create session on REST API: {e}")
        session_id = "sess_test_direct"

    # 2. Test WebSocket Connection
    ws_url = f"wss://emovision.onrender.com/ws/detection/{session_id}"
    print(f"[TEST] 2. Connecting to Production WebSocket: {ws_url}")

    try:
        import websockets
        async with websockets.connect(ws_url, ssl=True) as ws:
            print("[TEST] Handshake ACCEPTED! Connected to WebSocket.")
            
            # Create a 320x240 dummy test image with a circle
            dummy_img = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.circle(dummy_img, (160, 120), 50, (255, 255, 255), -1)
            _, buffer = cv2.imencode('.jpg', dummy_img)
            b64_frame = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

            print("[TEST] 3. Sending test video frame payload...")
            await ws.send(b64_frame)
            print("[TEST] Frame SENT.")

            print("[TEST] 4. Awaiting response JSON from Render...")
            response_text = await asyncio.wait_for(ws.recv(), timeout=10.0)
            print(f"[TEST SUCCESS] Response received from Render:")
            print(response_text)

    except Exception as e:
        import traceback
        print(f"[TEST ERROR] WebSocket test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_production_websocket())
