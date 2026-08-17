"""
Image & Asynchronous Video Upload Analysis Router for Emovision Backend.
Integrates SCRFD Multi-Face Detection and EfficientFace Batch Inference for file uploads.

Endpoints:
- POST /api/analyze/image                  : Synchronous image FER analysis
- POST /api/analyze/video                  : Asynchronous background video FER job submit
- GET  /api/analyze/video/{analysis_id}/status: Query real frame-progress status
- GET  /api/analyze/video/{analysis_id}/result: Query completed video FER results
- GET  /api/analyze/video/active           : Query currently running or latest active video job
- GET  /api/analysis/{analysis_id}          : Unified session analysis lookup
"""
import os
import uuid
import time
import math
import base64
import tempfile
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, UploadFile, File, HTTPException, status
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

# Thread pool for background video processing
bg_executor = ThreadPoolExecutor(max_workers=4)

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".webm", ".mkv"}

class VideoJobStatus:
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# In-memory status store for background video jobs: analysis_id -> Job Dict
_video_jobs: Dict[str, Dict[str, Any]] = {}

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
    Stores session record in database (source_type='image') and returns results immediately.
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

        analysis_id = f"img_{uuid.uuid4().hex[:8]}"

        raw_dets = detector.detect_faces(frame)
        
        if not raw_dets:
            # Persist 0-face image record in database
            try:
                db_repo.create_session(session_id=analysis_id, session_name=f"📷 {file.filename}", source_type="image")
                db_repo.end_session(session_id=analysis_id, total_frames=1, avg_fps=1.0)
            except Exception:
                pass

            return {
                "success": False,
                "message": "No face detected in this image.",
                "analysis_id": analysis_id,
                "analysis_type": "image",
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

        # Persist session record in Supabase / SQLite database
        try:
            db_repo.create_session(session_id=analysis_id, session_name=f"📷 {file.filename}", source_type="image")
            # Log frame predictions
            db_repo.log_frame_predictions(
                session_id=analysis_id,
                frame_number=1,
                detections=[
                    {
                        "person_id": idx,
                        "bbox": d["bbox"],
                        "emotion": d["emotion"],
                        "emotion_confidence": d["confidence"]
                    } for idx, d in enumerate(classified_dets, start=1)
                ]
            )
            db_repo.end_session(session_id=analysis_id, total_frames=1, avg_fps=1.0)
        except Exception as db_err:
            print(f"[Image Analysis DB Notice] Could not log image session to database: {db_err}")

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

def _process_video_job_bg(analysis_id: str, temp_path: str, filename: str):
    """Background worker processing video frames asynchronously."""
    try:
        job = _video_jobs.get(analysis_id)
        if not job:
            return

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            job["status"] = VideoJobStatus.FAILED
            job["error_message"] = "Could not open uploaded video file."
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = round(total_frames / fps, 2) if fps > 0 else 0.0

        sample_step = max(1, int(round(fps / 5.0)))
        total_frames_to_process = max(1, math.ceil(total_frames / sample_step))

        job["total_frames_to_process"] = total_frames_to_process

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
                job["frames_processed"] = analyzed_frames
                job["progress"] = min(99, int((analyzed_frames / total_frames_to_process) * 100))

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

        avg_conf = round(total_conf / total_detections_count, 4) if total_detections_count > 0 else 0.0
        dominant_emotion = max(overall_emotion_counts, key=overall_emotion_counts.get) if overall_emotion_counts else "Uncertain"

        dist = {}
        for em, cnt in overall_emotion_counts.items():
            dist[em] = {
                "count": cnt,
                "percentage": round(cnt / total_detections_count * 100.0, 1)
            }

        if total_detections_count == 0:
            result = {
                "success": False,
                "message": "No faces detected in this video.",
                "analysis_id": analysis_id,
                "analysis_type": "video",
                "filename": filename,
                "video_duration_seconds": duration_sec,
                "total_frames_analyzed": analyzed_frames,
                "total_face_detections": 0,
                "dominant_emotion": None,
                "average_confidence": 0.0,
                "emotion_distribution": {},
                "timeline": []
            }
        else:
            result = {
                "success": True,
                "analysis_id": analysis_id,
                "analysis_type": "video",
                "filename": filename,
                "video_duration_seconds": duration_sec,
                "total_frames_analyzed": analyzed_frames,
                "total_face_detections": total_detections_count,
                "dominant_emotion": dominant_emotion,
                "average_confidence": avg_conf,
                "emotion_distribution": dist,
                "timeline": timeline
            }

        job["status"] = VideoJobStatus.COMPLETED
        job["progress"] = 100
        job["result"] = result

        # Persist completed video session into database
        try:
            db_repo.end_session(session_id=analysis_id, total_frames=analyzed_frames, avg_fps=5.0)
        except Exception as db_err:
            print(f"[Video Analysis DB Notice] Could not update video session in database: {db_err}")

    except Exception as err:
        job = _video_jobs.get(analysis_id)
        if job:
            job["status"] = VideoJobStatus.FAILED
            job["error_message"] = str(err)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@router.post("/analyze/video")
async def start_analyze_video(file: UploadFile = File(...)):
    """
    Submits a video for asynchronous FER analysis.
    Returns immediately with analysis_id and status='processing'.
    Processing continues in the background surviving browser tab changes.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty video filename uploaded.")
        
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video file format '{ext}'. Allowed formats: {', '.join(ALLOWED_VIDEO_EXTS)}"
        )

    temp_dir = tempfile.gettempdir()
    analysis_id = f"vid_{uuid.uuid4().hex[:8]}"
    temp_path = os.path.join(temp_dir, f"{analysis_id}{ext}")

    try:
        contents = await file.read()
        if not contents or len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty video file uploaded.")

        with open(temp_path, "wb") as f:
            f.write(contents)

        # Initialize background job state
        _video_jobs[analysis_id] = {
            "analysis_id": analysis_id,
            "status": VideoJobStatus.PROCESSING,
            "filename": file.filename,
            "progress": 0,
            "frames_processed": 0,
            "total_frames_to_process": 1,
            "result": None,
            "error_message": None,
            "created_at": datetime.utcnow().isoformat()
        }

        # Create session record in database (source_type='video')
        try:
            db_repo.create_session(session_id=analysis_id, session_name=f"🎥 {file.filename}", source_type="video")
        except Exception as db_err:
            print(f"[Video Analysis DB Notice] Could not create video session: {db_err}")

        # Submit background worker
        bg_executor.submit(_process_video_job_bg, analysis_id, temp_path, file.filename)

        return {
            "success": True,
            "analysis_id": analysis_id,
            "status": VideoJobStatus.PROCESSING,
            "progress": 0,
            "message": "Video analysis job started successfully."
        }

    except HTTPException:
        raise
    except Exception as err:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Could not initialize video job: {str(err)}")

@router.get("/analyze/video/{analysis_id}/status")
def get_video_status(analysis_id: str):
    """
    Polls the current real progress of a background video processing job.
    Returns status ('processing', 'completed', 'failed'), progress (0..100), and frames_processed.
    """
    job = _video_jobs.get(analysis_id)
    if not job:
        # Check database if server restarted or job finished
        sess = db_repo.get_session(analysis_id)
        if sess:
            return {
                "analysis_id": analysis_id,
                "status": VideoJobStatus.COMPLETED,
                "progress": 100,
                "frames_processed": sess.get("total_frames", 0),
                "total_frames_to_process": sess.get("total_frames", 0),
                "error_message": None
            }
        raise HTTPException(status_code=404, detail=f"Video analysis job '{analysis_id}' not found.")

    return {
        "analysis_id": job["analysis_id"],
        "status": job["status"],
        "progress": job["progress"],
        "frames_processed": job["frames_processed"],
        "total_frames_to_process": job["total_frames_to_process"],
        "error_message": job["error_message"]
    }

@router.get("/analyze/video/{analysis_id}/result")
def get_video_result(analysis_id: str):
    """
    Fetches the final completed FER analysis dictionary for a video job.
    """
    job = _video_jobs.get(analysis_id)
    if job and job["result"] is not None:
        return job["result"]

    # Check database fallback
    sess_analytics = db_repo.get_session_analytics(analysis_id)
    sess_info = db_repo.get_session(analysis_id)
    if sess_info:
        return {
            "success": True if (sess_info.get("total_people_detected", 0) > 0) else False,
            "analysis_id": analysis_id,
            "analysis_type": "video",
            "filename": sess_info.get("session_name", "").replace("🎥 ", ""),
            "video_duration_seconds": sess_info.get("duration_seconds", 0.0),
            "total_frames_analyzed": sess_info.get("total_frames", 0),
            "total_face_detections": sess_info.get("total_predictions", 0),
            "dominant_emotion": sess_info.get("dominant_expression", "Neutral"),
            "average_confidence": sess_info.get("avg_confidence", 0.0),
            "emotion_distribution": sess_analytics.get("expression_distribution", {}) if sess_analytics else {},
            "timeline": []
        }

    raise HTTPException(status_code=404, detail=f"Results for analysis '{analysis_id}' not ready or missing.")

@router.get("/analyze/video/active")
def get_active_video_job():
    """
    Queries any currently running background video job to survive page refreshes and tab switches.
    """
    for aid, job in reversed(list(_video_jobs.items())):
        if job["status"] == VideoJobStatus.PROCESSING:
            return {
                "active_job": True,
                "analysis_id": aid,
                "status": job["status"],
                "progress": job["progress"],
                "filename": job["filename"]
            }
            
    return {"active_job": False, "analysis_id": None}

@router.get("/analysis/{analysis_id}")
def get_unified_analysis(analysis_id: str):
    """Unified endpoint to retrieve session analysis info for webcam, image, or video."""
    sess = db_repo.get_session(analysis_id)
    if not sess:
        raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' not found.")
        
    analytics = db_repo.get_session_analytics(analysis_id)
    return {
        "session_info": sess,
        "analytics": analytics
    }
