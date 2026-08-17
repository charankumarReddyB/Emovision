"""
Image & Video Analysis API Endpoints for Emovision Backend.
Integrates SCRFD Multi-Face Detection and EfficientFace Batch Inference for file uploads.
Endpoints:
- POST /api/analyze/image
- POST /api/analyze/video
- GET  /api/analysis/{analysis_id}
"""
import os
import uuid
import time
import base64
import tempfile
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Query
from pydantic import BaseModel

from app.services.scrfd_detector import SCRFDDetector
from app.services.emotion_classifier import EmotionClassifier
from app.services.face_aligner import FaceAligner
from app.db.repository import get_db_repository

router = APIRouter(prefix="/api", tags=["Analysis"])

# Instantiate dedicated detector and classifier instances for upload paths
detector = SCRFDDetector(score_threshold=0.30, nms_threshold=0.35)
classifier = EmotionClassifier()
aligner = FaceAligner(target_size=(224, 224))
db_repo = get_db_repository()

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".webm", ".mkv"}

def draw_annotated_faces(frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """Draws green bounding boxes, facial keypoints, and emotion labels onto a frame copy."""
    vis = frame.copy()
    for idx, det in enumerate(detections, start=1):
        x, y, w, h = det["bbox"]
        label = det.get("emotion", "Uncertain")
        conf = det.get("confidence", 0.0)
        
        color = (0, 255, 0)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        
        text = f"#{idx} {label} {conf*100:.0f}%"
        cv2.rectangle(vis, (x, max(0, y - 24)), (x + len(text) * 11, max(0, y)), (0, 0, 0), -1)
        cv2.putText(vis, text, (x + 4, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        
        kps = det.get("kps")
        if kps is not None:
            for kp in kps:
                cv2.circle(vis, (int(kp[0]), int(kp[1])), 3, (0, 0, 255), -1)
                
    return vis

@router.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyzes an uploaded image (JPG, PNG, WEBP) for multi-face expression recognition.
    Returns detected faces, bounding boxes, Base64 annotated image, and summary statistics.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename uploaded.")
        
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image file format '{ext}'. Allowed formats: {', '.join(ALLOWED_IMAGE_EXTS)}"
        )

    try:
        contents = await file.read()
        if not contents or len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty image file uploaded.")
            
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None or frame.size == 0:
            raise HTTPException(status_code=400, detail="Corrupted or invalid image file.")

        raw_dets = detector.detect_faces(frame)
        
        if not raw_dets:
            return {
                "success": False,
                "message": "No face detected in this image.",
                "filename": file.filename,
                "total_faces": 0,
                "dominant_emotion": None,
                "average_confidence": 0.0,
                "emotion_distribution": {},
                "annotated_image_base64": None,
                "detections": []
            }

        aligned_chips = []
        bboxes = []
        for det in raw_dets:
            bbox = det["bbox"]
            kps = det.get("kps")
            chip = aligner.align_face(frame, kps=kps, bbox=bbox)
            aligned_chips.append(chip)
            bboxes.append(bbox)

        emotions_list = classifier.classify_batch(aligned_chips, bboxes=bboxes)

        classified_dets = []
        emotion_counts: Dict[str, int] = {}
        total_conf = 0.0

        for idx, (det, (emotion, conf)) in enumerate(zip(raw_dets, emotions_list), start=1):
            total_conf += conf
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            det_info = {
                "face_index": idx,
                "bbox": det["bbox"],
                "emotion": emotion,
                "confidence": round(conf, 4)
            }
            classified_dets.append(det_info)

        annotated_frame = draw_annotated_faces(frame, classified_dets)
        _, buffer = cv2.imencode(".jpg", annotated_frame)
        base64_str = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

        total_faces = len(classified_dets)
        avg_conf = round(total_conf / total_faces, 4) if total_faces > 0 else 0.0
        dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "Uncertain"

        dist = {}
        for em, cnt in emotion_counts.items():
            dist[em] = {
                "count": cnt,
                "percentage": round(cnt / total_faces * 100.0, 1)
            }

        analysis_id = f"img_{uuid.uuid4().hex[:8]}"
        try:
            db_repo.create_session(session_id=analysis_id, session_name=f"Image: {file.filename}", source_type="image")
            db_repo.end_session(session_id=analysis_id, total_frames=1, avg_fps=1.0)
        except Exception:
            pass

        return {
            "success": True,
            "analysis_id": analysis_id,
            "analysis_type": "image",
            "filename": file.filename,
            "total_faces": total_faces,
            "dominant_emotion": dominant_emotion,
            "average_confidence": avg_conf,
            "emotion_distribution": dist,
            "annotated_image_base64": base64_str,
            "detections": classified_dets
        }

    except HTTPException:
        raise
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Image analysis error: {str(err)}")

@router.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    """
    Analyzes an uploaded video (MP4, AVI, MOV, WEBM) using frame sampling & single-pass EfficientFace batching.
    Returns total frames, face count, video duration, emotion timeline, and aggregated distribution.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty video filename uploaded.")
        
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video file format '{ext}'. Allowed formats: {', '.join(ALLOWED_VIDEO_EXTS)}"
        )

    # Save uploaded video to temporary file
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"video_{uuid.uuid4().hex[:8]}{ext}")

    try:
        contents = await file.read()
        if not contents or len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty video file uploaded.")

        with open(temp_path, "wb") as f:
            f.write(contents)

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file or file is corrupted.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = round(total_frames / fps, 2) if fps > 0 else 0.0

        # Frame sampling: sample approximately 5 frames per second
        sample_step = max(1, int(round(fps / 5.0)))

        frame_idx = 0
        analyzed_frames = 0
        total_detections_count = 0
        total_conf = 0.0

        overall_emotion_counts: Dict[str, int] = {}
        timeline = []

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % sample_step == 0:
                analyzed_frames += 1
                timestamp_sec = round(frame_idx / fps, 2)

                raw_dets = detector.detect_faces(frame)
                
                if raw_dets:
                    aligned_chips = []
                    bboxes = []
                    for det in raw_dets:
                        chip = aligner.align_face(frame, kps=det.get("kps"), bbox=det["bbox"])
                        aligned_chips.append(chip)
                        bboxes.append(det["bbox"])

                    emotions_list = classifier.classify_batch(aligned_chips, bboxes=bboxes)

                    frame_emotions = []
                    for det, (emotion, conf) in zip(raw_dets, emotions_list):
                        total_detections_count += 1
                        total_conf += conf
                        overall_emotion_counts[emotion] = overall_emotion_counts.get(emotion, 0) + 1
                        frame_emotions.append({
                            "bbox": det["bbox"],
                            "emotion": emotion,
                            "confidence": round(conf, 4)
                        })

                    frame_dominant = max(
                        set([e["emotion"] for e in frame_emotions]),
                        key=lambda em: [e["emotion"] for e in frame_emotions].count(em)
                    )

                    timeline.append({
                        "timestamp_sec": timestamp_sec,
                        "frame_idx": frame_idx,
                        "face_count": len(frame_emotions),
                        "dominant_emotion": frame_dominant,
                        "detections": frame_emotions
                    })

            frame_idx += 1

        cap.release()

        if total_detections_count == 0:
            return {
                "success": False,
                "message": "No faces detected in this video.",
                "filename": file.filename,
                "video_duration_seconds": duration_sec,
                "total_frames_analyzed": analyzed_frames,
                "total_face_detections": 0
            }

        avg_conf = round(total_conf / total_detections_count, 4) if total_detections_count > 0 else 0.0
        dominant_emotion = max(overall_emotion_counts, key=overall_emotion_counts.get) if overall_emotion_counts else "Uncertain"

        dist = {}
        for em, cnt in overall_emotion_counts.items():
            dist[em] = {
                "count": cnt,
                "percentage": round(cnt / total_detections_count * 100.0, 1)
            }

        analysis_id = f"vid_{uuid.uuid4().hex[:8]}"
        try:
            db_repo.create_session(session_id=analysis_id, session_name=f"Video: {file.filename}", source_type="video")
            db_repo.end_session(session_id=analysis_id, total_frames=analyzed_frames, avg_fps=5.0)
        except Exception:
            pass

        return {
            "success": True,
            "analysis_id": analysis_id,
            "analysis_type": "video",
            "filename": file.filename,
            "video_duration_seconds": duration_sec,
            "total_frames_analyzed": analyzed_frames,
            "total_face_detections": total_detections_count,
            "dominant_emotion": dominant_emotion,
            "average_confidence": avg_conf,
            "emotion_distribution": dist,
            "timeline": timeline
        }

    except HTTPException:
        raise
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Video analysis error: {str(err)}")
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
