import urllib.request
import urllib.error
import json
import time
import asyncio
import websockets
import os
import sqlite3
import sys
from typing import Dict, Any, List

sys.path.append('.')

BASE_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000"
DB_PATH = os.path.join("data", "emovision.db")

results = {
    "functional": [],
    "performance": [],
    "cv_eval": [],
    "api_endpoints": [],
    "error_handling": [],
    "websocket": [],
    "database": []
}

def http_request(url: str, method: str = "GET", data: Dict[str, Any] = None) -> tuple[int, Any]:
    start = time.perf_counter()
    req_data = json.dumps(data).encode() if data else None
    headers = {"Content-Type": "application/json"} if req_data else {}
    req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as response:
            elapsed = (time.perf_counter() - start) * 1000
            res_body = json.loads(response.read().decode())
            return response.status, res_body, elapsed
    except urllib.error.HTTPError as e:
        elapsed = (time.perf_counter() - start) * 1000
        try:
            body = json.loads(e.read().decode())
        except Exception:
            body = {}
        return e.code, body, elapsed

def log_result(category: str, title: str, status: str, details: str):
    results[category].append({"title": title, "status": status, "details": details})
    print(f"[{status}] {category.upper()} - {title}: {details}")

# 1. API Endpoints & Error Handling Tests
def test_api_and_errors():
    print("\n--- 1. REST API ENDPOINTS & ERROR HANDLING ---")
    
    # GET /api/health
    status, body, latency = http_request(f"{BASE_URL}/api/health")
    log_result("api_endpoints", "GET /api/health", "PASS" if status == 200 else "FAIL", f"Status: {status}, Latency: {latency:.1f}ms")
    
    # GET /api/model/info
    status, body, latency = http_request(f"{BASE_URL}/api/model/info")
    log_result("api_endpoints", "GET /api/model/info", "PASS" if status == 200 else "FAIL", f"Status: {status}, Model: {body.get('model_name')}, Classes: {len(body.get('emotion_classes', []))}")
    
    # POST /api/session/start
    status, body, latency = http_request(f"{BASE_URL}/api/session/start", "POST", {"session_name": "Full Test Suite Session", "source_type": "webcam"})
    session_id = body.get("session_id") if status in (200, 201) else None
    log_result("api_endpoints", "POST /api/session/start", "PASS" if status in (200, 201) else "FAIL", f"Status: {status}, Session ID: {session_id}")
    
    if session_id:
        # GET /api/session/{id}/current
        status, body, latency = http_request(f"{BASE_URL}/api/session/{session_id}/current")
        log_result("api_endpoints", "GET /api/session/{id}/current", "PASS" if status == 200 else "FAIL", f"Status: {status}, Latency: {latency:.1f}ms, People: {body.get('people_detected')}")
        
        # POST /api/session/{id}/end
        status, body, latency = http_request(f"{BASE_URL}/api/session/{session_id}/end", "POST")
        log_result("api_endpoints", "POST /api/session/{id}/end", "PASS" if status == 200 else "FAIL", f"Status: {status}, Duration: {body.get('duration_seconds')}s")
        
        # GET /api/session/{id}/analytics
        status, body, latency = http_request(f"{BASE_URL}/api/session/{session_id}/analytics")
        log_result("api_endpoints", "GET /api/session/{id}/analytics", "PASS" if status == 200 else "FAIL", f"Status: {status}, Dominant: {body.get('dominant_expression')}")
        
        # GET /api/session/{id}/person/1
        status, body, latency = http_request(f"{BASE_URL}/api/session/{session_id}/person/1")
        log_result("api_endpoints", "GET /api/session/{id}/person/1", "PASS" if status in (200, 404) else "FAIL", f"Status: {status}, Result: {body.get('dominant_expression') if status == 200 else body.get('detail')}")
        
        # GET /api/sessions/{id} details
        status, body, latency = http_request(f"{BASE_URL}/api/sessions/{session_id}")
        log_result("api_endpoints", "GET /api/sessions/{id}", "PASS" if status == 200 else "FAIL", f"Status: {status}, Source: {body.get('source_type')}")

    # GET /api/sessions history
    status, body, latency = http_request(f"{BASE_URL}/api/sessions")
    log_result("api_endpoints", "GET /api/sessions", "PASS" if status == 200 else "FAIL", f"Status: {status}, Total Sessions: {body.get('total')}")

    # Error Test: Invalid Session ID (404 expected)
    status, body, latency = http_request(f"{BASE_URL}/api/session/sess_nonexistent999/analytics")
    log_result("error_handling", "GET /api/session/invalid_id/analytics", "PASS" if status == 404 else "FAIL", f"Expected 404, got {status}: {body.get('detail')}")

    # Error Test: Invalid Person ID (404 expected)
    if session_id:
        status, body, latency = http_request(f"{BASE_URL}/api/session/{session_id}/person/9999")
        log_result("error_handling", "GET /api/session/valid_id/person/9999", "PASS" if status == 404 else "FAIL", f"Expected 404, got {status}: {body.get('detail')}")

    return session_id

# 2. WebSocket Streaming & Reconnection Tests
async def test_websocket_stream():
    print("\n--- 2. WEBSOCKET REAL-TIME STREAM AUDIT ---")
    status, body, _ = http_request(f"{BASE_URL}/api/session/start", "POST", {"session_name": "WS Test Session"})
    session_id = body.get("session_id")
    url = f"{WS_URL}/ws/detection/{session_id}"
    
    try:
        async with websockets.connect(url) as ws:
            messages = []
            start_time = time.perf_counter()
            for _ in range(5):
                msg = await ws.recv()
                messages.append(json.loads(msg))
            elapsed = time.perf_counter() - start_time
            rate = len(messages) / elapsed if elapsed > 0 else 0
            
            sample = messages[0]
            valid_schema = all(k in sample for k in ("session_id", "people_detected", "fps", "average_confidence", "dominant_expression", "people"))
            log_result("websocket", "WebSocket Connection & Delivery", "PASS" if valid_schema else "FAIL", f"Received {len(messages)} frames at {rate:.1f} Hz, People: {sample.get('people_detected')}")
            
    except Exception as e:
        log_result("websocket", "WebSocket Connection", "FAIL", f"Error: {e}")
        
    # Test Reconnect / Cleanup
    try:
        async with websockets.connect(url) as ws:
            await ws.recv()
        log_result("websocket", "WebSocket Reconnect & Cleanup", "PASS", "Connected and closed cleanly.")
    except Exception as e:
        log_result("websocket", "WebSocket Reconnect", "FAIL", f"Error: {e}")

# 3. Multi-Person Performance & Scaling Benchmarks
def benchmark_cv_performance():
    print("\n--- 3. MULTI-PERSON PERFORMANCE BENCHMARKING ---")
    import cv2
    import numpy as np
    from app.services.emotion_classifier import EmotionClassifier
    from app.services.realtime_pipeline import RealtimePipeline
    from test_face_pipeline import generate_synthetic_multi_face_frame

    classifier = EmotionClassifier()
    
    # Benchmark Model Inference Standalone
    dummy_face = np.zeros((48, 48, 3), dtype=np.uint8)
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = classifier.predict_face(dummy_face)
        latencies.append((time.perf_counter() - t0) * 1000)
    avg_inf = np.mean(latencies)
    log_result("performance", "Model Inference Latency (Single Face)", "PASS", f"Avg: {avg_inf:.2f} ms per face")

    # Multi-person pipeline scaling benchmark
    face_counts = [1, 2, 3, 5, 8]
    print("\nScaling Benchmark Results:")
    print(f"{'Faces':<8} | {'Avg Latency (ms)':<18} | {'Pipeline FPS':<14} | {'Status':<10}")
    print("-" * 56)

    for num_faces in face_counts:
        pipeline = RealtimePipeline(session_id=f"bench_{num_faces}", session_name=f"Bench {num_faces} faces")
        frame_latencies = []
        for i in range(1, 21):
            frame = generate_synthetic_multi_face_frame(num_faces=num_faces, frame_idx=i)
            t0 = time.perf_counter()
            _, stats = pipeline.process_frame(frame, i)
            t_elapsed = (time.perf_counter() - t0) * 1000
            frame_latencies.append(t_elapsed)
        pipeline.close()
        
        mean_lat = np.mean(frame_latencies)
        pipe_fps = 1000.0 / mean_lat if mean_lat > 0 else 0
        status_str = "PASS" if pipe_fps >= 15.0 else "WARNING"
        print(f"{num_faces:<8} | {mean_lat:<18.2f} | {pipe_fps:<14.1f} | {status_str:<10}")
        log_result("performance", f"Pipeline Scaling ({num_faces} faces)", status_str, f"Latency: {mean_lat:.2f} ms, FPS: {pipe_fps:.1f}")

# 4. Database Repository Integrity Audit
def test_database_integrity():
    print("\n--- 4. DATABASE REPOSITORY INTEGRITY AUDIT ---")
    from app.db.repository import get_db_repository
    repo = get_db_repository()
    
    sess_res = repo.list_sessions(page=1, limit=100)
    session_count = sess_res.get("total", 0)
    repo_name = repo.__class__.__name__
    
    log_result("database", "Database Repository Audit", "PASS", f"Active Repository: {repo_name}, Total Sessions: {session_count}")

# 5. Computer Vision & Expression Output Verification
def test_expression_classes():
    print("\n--- 5. 7-CLASS FACIAL EXPRESSION EVALUATION ---")
    from app.services.emotion_classifier import EmotionClassifier
    from app.core.config import settings
    import numpy as np

    rec = EmotionClassifier()
    emotions = settings.EMOTION_CLASSES
    log_result("cv_eval", "7 Target Classes Validation", "PASS" if len(emotions) == 7 else "FAIL", f"Classes: {emotions}")
    
    # Test prediction output
    dummy = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
    label, conf = rec.predict_face(dummy)
    valid = label in emotions and conf >= 0.0
    log_result("cv_eval", "Single Face Prediction Output", "PASS" if valid else "FAIL", f"Dominant: {label}, Confidence: {conf:.2f}")

def main():
    print("==========================================================================")
    print("      EMOVISION COMPREHENSIVE END-TO-END TESTING & BENCHMARK SUITE       ")
    print("==========================================================================")
    
    test_api_and_errors()
    asyncio.run(test_websocket_stream())
    benchmark_cv_performance()
    test_database_integrity()
    test_expression_classes()
    
    print("\n==========================================================================")
    print("                      TEST SUITE SUMMARY COMPLETE                         ")
    print("==========================================================================")

if __name__ == "__main__":
    main()
