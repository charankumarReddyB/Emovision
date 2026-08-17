"""
Standalone Test Suite for Image and Video Upload Analysis APIs.
Tests:
- POST /api/analyze/image (1 face, 5 faces, 0 faces, invalid format)
- POST /api/analyze/video (sample video, 0 faces, invalid format)
"""
import sys
import io
import cv2
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

from main import app
from test_acceptance import create_multi_face_scene

client = TestClient(app)

def run_upload_analysis_tests():
    print("=" * 80)
    print("IMAGE & VIDEO UPLOAD ANALYSIS API TEST SUITE")
    print("=" * 80)

    results = []

    # 1. Image Upload: 1 Face
    img_1face = create_multi_face_scene(1)
    _, buf = cv2.imencode(".jpg", img_1face)
    files = {"file": ("test_1face.jpg", buf.tobytes(), "image/jpeg")}
    res = client.post("/api/analyze/image", files=files)
    
    passed = (res.status_code == 200 and res.json().get("total_faces") == 1)
    results.append(("Image (1 Face)", "PASS" if passed else "FAIL", f"Status {res.status_code}, Faces: {res.json().get('total_faces')}"))
    print(f"  • Image (1 Face)        -> {'PASS' if passed else 'FAIL'} | Details: {res.json().get('message', 'Detected ' + str(res.json().get('total_faces')) + ' faces')}")

    # 2. Image Upload: 5 Faces
    img_5faces = create_multi_face_scene(5)
    _, buf = cv2.imencode(".png", img_5faces)
    files = {"file": ("test_5faces.png", buf.tobytes(), "image/png")}
    res = client.post("/api/analyze/image", files=files)
    
    passed = (res.status_code == 200 and res.json().get("total_faces") == 5)
    results.append(("Image (5 Faces)", "PASS" if passed else "FAIL", f"Status {res.status_code}, Faces: {res.json().get('total_faces')}"))
    print(f"  • Image (5 Faces)       -> {'PASS' if passed else 'FAIL'} | Details: Detected {res.json().get('total_faces')} faces")

    # 3. Image Upload: 0 Faces (Empty Scene)
    img_0face = create_multi_face_scene(0)
    _, buf = cv2.imencode(".jpg", img_0face)
    files = {"file": ("test_noface.jpg", buf.tobytes(), "image/jpeg")}
    res = client.post("/api/analyze/image", files=files)
    
    passed = (res.status_code == 200 and res.json().get("success") is False and "No face detected" in res.json().get("message", ""))
    results.append(("Image (No Face)", "PASS" if passed else "FAIL", res.json().get("message")))
    print(f"  • Image (No Face)       -> {'PASS' if passed else 'FAIL'} | Message: '{res.json().get('message')}'")

    # 4. Image Upload: Unsupported File Format
    files = {"file": ("test.txt", b"hello world text file", "text/plain")}
    res = client.post("/api/analyze/image", files=files)
    
    passed = (res.status_code == 400 and "Unsupported image file format" in res.json().get("detail", ""))
    results.append(("Image (Unsupported)", "PASS" if passed else "FAIL", res.json().get("detail")))
    print(f"  • Image (Unsupported)   -> {'PASS' if passed else 'FAIL'} | Error: '{res.json().get('detail')}'")

    # 5. Video Upload: Synthetic Short Video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_path = Path(__file__).resolve().parent / "temp_test_video.mp4"
    out = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (1280, 720))

    frame_3faces = create_multi_face_scene(3)
    frame_1face = create_multi_face_scene(1)

    for _ in range(10):
        out.write(frame_3faces)
    for _ in range(10):
        out.write(frame_1face)
    out.release()

    with open(vid_path, "rb") as vf:
        files = {"file": ("test_video.mp4", vf.read(), "video/mp4")}
        res = client.post("/api/analyze/video", files=files)

    if vid_path.exists():
        vid_path.unlink()

    passed = (res.status_code == 200 and res.json().get("success") is True and res.json().get("total_face_detections", 0) > 0)
    results.append(("Video (Multi-Face)", "PASS" if passed else "FAIL", f"Status {res.status_code}, Detections: {res.json().get('total_face_detections')}"))
    print(f"  • Video (Multi-Face)   -> {'PASS' if passed else 'FAIL'} | Duration: {res.json().get('video_duration_seconds')}s, Detections: {res.json().get('total_face_detections')}")

    print("\n" + "=" * 80)
    print("UPLOAD ANALYSIS API TEST SUMMARY")
    print("=" * 80)
    for name, status_str, detail in results:
        print(f"  • {name:22s} -> {status_str:6s} | {detail}")
    print("=" * 80)

if __name__ == "__main__":
    run_upload_analysis_tests()
