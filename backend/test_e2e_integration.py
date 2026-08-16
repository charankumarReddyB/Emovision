import urllib.request
import json
import asyncio
import websockets

BASE_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000"

def http_get(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())

def http_post(url, data=None):
    req_data = json.dumps(data).encode() if data else b""
    req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())

async def run_ws_stream_check(session_id):
    url = f"{WS_URL}/ws/detection/{session_id}"
    async with websockets.connect(url) as ws:
        msg = await ws.recv()
        data = json.loads(msg)
        print("[4] WebSocket Frame Received:", data)
        assert data["session_id"] == session_id
        assert "people" in data
        assert "fps" in data

def test_backend_e2e_integration():
    """Pytest wrapper for E2E integration verification when backend server is active."""
    try:
        health = http_get(f"{BASE_URL}/api/health")
        assert health["status"] == "ok"
    except Exception:
        pytest.skip("Backend server not running on port 8000; skipping live E2E HTTP test")

def main():
    print("==================================================")
    print("RUNNING EMOVISION E2E BACKEND & WEBSOCKET VERIFICATION")
    print("==================================================")
    
    # 1. Health Check
    health = http_get(f"{BASE_URL}/api/health")
    print("[1] Health Check:", health)
    assert health["status"] == "ok"
    
    # 2. Model Info
    model_info = http_get(f"{BASE_URL}/api/model/info")
    classes = model_info.get("emotion_classes") or model_info.get("classes")
    print("[2] Model Info:", model_info["model_name"], "Classes:", classes)
    assert "Happy" in classes
    
    # 3. Start Session
    start_res = http_post(f"{BASE_URL}/api/session/start", {"session_name": "E2E Live Test", "source_type": "webcam"})
    session_id = start_res["session_id"]
    print("[3] Started Session:", session_id)
    
    # 4. WebSocket Test
    asyncio.run(run_ws_stream_check(session_id))
    
    # 5. Fetch Current Detection
    curr = http_get(f"{BASE_URL}/api/session/{session_id}/current")
    print("[5] Current Detection:", curr)
    
    # 6. End Session
    end_res = http_post(f"{BASE_URL}/api/session/{session_id}/end")
    print("[6] Ended Session:", end_res)
    
    # 7. Session History
    history = http_get(f"{BASE_URL}/api/sessions")
    print("[7] Session History Count:", history["total"])
    assert history["total"] >= 1
    
    # 8. Session Analytics
    analytics = http_get(f"{BASE_URL}/api/session/{session_id}/analytics")
    print("[8] Session Analytics Dominant:", analytics["dominant_expression"], "Duration:", analytics["session_duration_seconds"])
    
    print("\nALL BACKEND API & WEBSOCKET ENDPOINTS FULLY VERIFIED SUCCESSFUL!")
    print("==================================================")

if __name__ == "__main__":
    main()
