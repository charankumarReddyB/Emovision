"""
Database Access Layer Repository Pattern for Emovision Backend.
Provides abstract interface and implementations for both Supabase PostgreSQL (Production)
and local SQLite (Fallback).
"""
import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import Counter

from app.core.config import settings
from app.db.orm_models import Base, SessionModel, DetectionLogModel

# Import Supabase SDK safely
try:
    from supabase import create_client, Client
    HAS_SUPABASE_SDK = True
except ImportError:
    HAS_SUPABASE_SDK = False
    Client = Any

# Import SQLAlchemy for SQLite fallback
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker, Session

# -------------------------------------------------------------------------
# Base Abstract Repository Interface
# -------------------------------------------------------------------------
class DatabaseRepository(ABC):
    @abstractmethod
    def create_session(self, session_id: str, session_name: str = "Live Session", source_type: str = "webcam") -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def end_session(self, session_id: str, total_frames: int = 0, avg_fps: float = 0.0) -> Dict[str, Any]:
        pass

    @abstractmethod
    def log_frame_predictions(self, session_id: str, frame_number: int, detections: List[Dict[str, Any]]) -> bool:
        pass

    @abstractmethod
    def get_session_analytics(self, session_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_person_analytics(self, session_id: str, person_id: int) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_latest_frame_detections(self, session_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def list_sessions(self, page: int = 1, limit: int = 10) -> Dict[str, Any]:
        pass


# -------------------------------------------------------------------------
# Local SQLite Repository Implementation
# -------------------------------------------------------------------------
class SqliteRepository(DatabaseRepository):
    def __init__(self, db_path: str = None):
        path = db_path or str(settings.DATABASE_PATH)
        self.engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)

    def create_session(self, session_id: str, session_name: str = "Live Session", source_type: str = "webcam") -> Dict[str, Any]:
        db = self.SessionLocal()
        try:
            existing = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
            if existing:
                return {
                    "session_id": existing.session_id,
                    "session_name": existing.session_name,
                    "source_type": existing.source_type,
                    "start_time": existing.start_time.isoformat() if existing.start_time else "",
                    "status": existing.status
                }
            now = datetime.utcnow()
            sess = SessionModel(
                session_id=session_id,
                session_name=session_name,
                source_type=source_type,
                start_time=now,
                status="active"
            )
            db.add(sess)
            db.commit()
            db.refresh(sess)
            return {
                "session_id": sess.session_id,
                "session_name": sess.session_name,
                "source_type": sess.source_type,
                "start_time": sess.start_time.isoformat() if sess.start_time else "",
                "status": sess.status
            }
        finally:
            db.close()

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        db = self.SessionLocal()
        try:
            sess = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
            if not sess:
                return None
            total_logs = db.query(DetectionLogModel).filter(DetectionLogModel.session_id == session_id).count()
            return {
                "session_id": sess.session_id,
                "session_name": sess.session_name,
                "source_type": sess.source_type,
                "start_time": sess.start_time.isoformat() if sess.start_time else None,
                "end_time": sess.end_time.isoformat() if sess.end_time else None,
                "duration_seconds": sess.duration or 0.0,
                "total_frames": sess.total_frames or 0,
                "total_predictions": total_logs,
                "total_people_detected": sess.total_people_detected or 0,
                "avg_fps": sess.avg_fps or 0.0,
                "dominant_expression": sess.dominant_expression or "Neutral",
                "status": sess.status
            }
        finally:
            db.close()

    def end_session(self, session_id: str, total_frames: int = 0, avg_fps: float = 0.0) -> Dict[str, Any]:
        db = self.SessionLocal()
        try:
            sess = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
            if not sess:
                return {}
            now = datetime.utcnow()
            sess.end_time = now
            sess.status = "completed"
            if total_frames > 0:
                sess.total_frames = total_frames
            if avg_fps > 0:
                sess.avg_fps = avg_fps

            logs = db.query(DetectionLogModel).filter(DetectionLogModel.session_id == session_id).all()
            unique_pids = set(l.person_id for l in logs if l.person_id > 0)
            sess.total_predictions = len(logs)
            sess.total_people_detected = len(unique_pids)

            emotions = [l.emotion_label for l in logs if l.emotion_label]
            dominant_emo = Counter(emotions).most_common(1)[0][0] if emotions else "Neutral"
            sess.dominant_expression = dominant_emo

            confidences = [l.emotion_confidence for l in logs if l.emotion_confidence is not None]
            sess.avg_confidence = round(float(sum(confidences) / len(confidences)), 4) if confidences else 0.0

            db.commit()
            return {
                "session_id": sess.session_id,
                "start_time": sess.start_time.isoformat() if sess.start_time else "",
                "end_time": sess.end_time.isoformat(),
                "duration_seconds": sess.duration,
                "total_predictions": len(logs),
                "total_people_detected": len(unique_pids),
                "dominant_expression": dominant_emo
            }
        finally:
            db.close()

    def log_frame_predictions(self, session_id: str, frame_number: int, detections: List[Dict[str, Any]]) -> bool:
        db = self.SessionLocal()
        try:
            logs = []
            for det in detections:
                bbox = det.get("bbox", (0, 0, 0, 0))
                log = DetectionLogModel(
                    session_id=session_id,
                    frame_number=frame_number,
                    person_id=det.get("person_id", -1),
                    bbox_x=int(bbox[0]),
                    bbox_y=int(bbox[1]),
                    bbox_w=int(bbox[2]),
                    bbox_h=int(bbox[3]),
                    confidence=float(det.get("confidence", 1.0)),
                    emotion_label=det.get("emotion", "Neutral"),
                    emotion_confidence=float(det.get("emotion_confidence", 0.0))
                )
                logs.append(log)
            if logs:
                db.bulk_save_objects(logs)
                db.commit()
            return True
        except Exception as e:
            print(f"[SqliteRepository Error] Logging frame predictions failed: {e}")
            return False
        finally:
            db.close()

    def get_session_analytics(self, session_id: str) -> Optional[Dict[str, Any]]:
        db = self.SessionLocal()
        try:
            sess = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
            if not sess:
                return None
            logs = db.query(DetectionLogModel).filter(DetectionLogModel.session_id == session_id).all()
            if not logs:
                return {
                    "session_id": session_id,
                    "total_people_detected": 0,
                    "total_predictions": 0,
                    "expression_distribution": {},
                    "average_confidence": 0.0,
                    "dominant_expression": "None",
                    "session_duration_seconds": sess.duration or 0.0,
                    "avg_fps": sess.avg_fps or 0.0,
                    "persons": []
                }
            unique_pids = set(l.person_id for l in logs if l.person_id > 0)
            emotions = [l.emotion_label for l in logs if l.emotion_label]
            confidences = [l.emotion_confidence for l in logs if l.emotion_confidence is not None]
            distribution = dict(Counter(emotions))
            dominant_emo = max(distribution, key=distribution.get) if distribution else "Neutral"
            avg_conf = float(sum(confidences) / len(confidences)) * 100 if confidences else 0.0

            return {
                "session_id": session_id,
                "total_people_detected": len(unique_pids),
                "total_predictions": len(logs),
                "expression_distribution": distribution,
                "average_confidence": round(avg_conf, 1),
                "dominant_expression": dominant_emo,
                "session_duration_seconds": sess.duration or 0.0,
                "avg_fps": sess.avg_fps or 0.0,
                "persons": sorted(list(unique_pids))
            }
        finally:
            db.close()

    def get_person_analytics(self, session_id: str, person_id: int) -> Optional[Dict[str, Any]]:
        db = self.SessionLocal()
        try:
            sess = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
            if not sess:
                return None
            person_logs = db.query(DetectionLogModel)\
                            .filter(DetectionLogModel.session_id == session_id, DetectionLogModel.person_id == person_id)\
                            .order_by(DetectionLogModel.frame_number.asc())\
                            .all()
            if not person_logs:
                return None
            emotions = [l.emotion_label for l in person_logs if l.emotion_label]
            confidences = [l.emotion_confidence for l in person_logs if l.emotion_confidence is not None]
            distribution = dict(Counter(emotions))
            dominant_emo = max(distribution, key=distribution.get) if distribution else "Neutral"
            avg_conf = float(sum(confidences) / len(confidences)) * 100 if confidences else 0.0

            return {
                "person_id": person_id,
                "dominant_expression": dominant_emo,
                "average_confidence": round(avg_conf, 1),
                "expression_distribution": distribution,
                "expression_timeline": emotions
            }
        finally:
            db.close()

    def get_latest_frame_detections(self, session_id: str) -> Dict[str, Any]:
        db = self.SessionLocal()
        try:
            sess = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
            if not sess:
                return {}
            latest_frame = db.query(DetectionLogModel.frame_number)\
                             .filter(DetectionLogModel.session_id == session_id)\
                             .order_by(DetectionLogModel.frame_number.desc())\
                             .first()
            if not latest_frame:
                return {
                    "session_id": session_id,
                    "people_detected": 0,
                    "fps": sess.avg_fps or 0.0,
                    "average_confidence": 0.0,
                    "dominant_expression": "None",
                    "people": []
                }
            frame_num = latest_frame[0]
            frame_logs = db.query(DetectionLogModel)\
                           .filter(DetectionLogModel.session_id == session_id, DetectionLogModel.frame_number == frame_num)\
                           .all()
            people_list = []
            emotions = []
            confidences = []
            for l in frame_logs:
                emotions.append(l.emotion_label or "Neutral")
                confidences.append(l.emotion_confidence or 0.0)
                people_list.append({
                    "person_id": l.person_id,
                    "expression": l.emotion_label or "Neutral",
                    "confidence": round(float(l.emotion_confidence or 0.0), 2),
                    "bounding_box": {
                        "x": l.bbox_x,
                        "y": l.bbox_y,
                        "width": l.bbox_w,
                        "height": l.bbox_h
                    }
                })
            counts = Counter(emotions) if emotions else {}
            dom_emo = counts.most_common(1)[0][0] if counts else "None"
            avg_conf = float(sum(confidences) / len(confidences)) * 100 if confidences else 0.0
            return {
                "session_id": session_id,
                "people_detected": len(people_list),
                "fps": sess.avg_fps or 30.0,
                "average_confidence": round(avg_conf, 1),
                "dominant_expression": dom_emo,
                "people": people_list
            }
        finally:
            db.close()

    def list_sessions(self, page: int = 1, limit: int = 10) -> Dict[str, Any]:
        db = self.SessionLocal()
        try:
            total = db.query(SessionModel).count()
            offset = (page - 1) * limit
            sessions_raw = db.query(SessionModel)\
                             .order_by(SessionModel.start_time.desc())\
                             .offset(offset)\
                             .limit(limit)\
                             .all()
            sessions_list = []
            for s in sessions_raw:
                sessions_list.append({
                    "session_id": s.session_id,
                    "session_name": s.session_name or "Session",
                    "date": s.start_time.strftime("%Y-%m-%d") if s.start_time else "",
                    "duration_seconds": s.duration or 0.0,
                    "people_count": s.total_people_detected or 0,
                    "dominant_expression": s.dominant_expression or "Neutral",
                    "average_confidence": round((s.avg_confidence or 0.0) * 100, 1),
                    "status": s.status or "completed"
                })
            return {
                "total": total,
                "page": page,
                "limit": limit,
                "sessions": sessions_list
            }
        finally:
            db.close()


from concurrent.futures import ThreadPoolExecutor

# -------------------------------------------------------------------------
# Supabase Production PostgreSQL Repository Implementation
# -------------------------------------------------------------------------
class SupabaseRepository(DatabaseRepository):
    def __init__(self, url: str = None, key: str = None):
        self.url = url or settings.SUPABASE_URL
        self.key = key or settings.SUPABASE_KEY
        self.client: Optional[Client] = None
        self.executor = ThreadPoolExecutor(max_workers=4)

        if HAS_SUPABASE_SDK and self.url and self.key:
            try:
                self.client = create_client(self.url, self.key)
                print(f"[SupabaseRepository] Connected to Supabase client: {self.url}")
            except Exception as e:
                print(f"[SupabaseRepository Warning] Could not initialize Supabase SDK client: {e}")

        # Fallback to SqliteRepository if credentials incomplete
        self.fallback_repo = SqliteRepository() if not self.client else None

    def create_session(self, session_id: str, session_name: str = "Live Session", source_type: str = "webcam") -> Dict[str, Any]:
        if not self.client:
            return self.fallback_repo.create_session(session_id, session_name, source_type)

        now_iso = datetime.utcnow().isoformat()
        data = {
            "id": session_id,
            "session_name": session_name,
            "source_type": source_type,
            "started_at": now_iso,
            "status": "active"
        }
        res = self.client.table("sessions").upsert(data).execute()
        return {
            "session_id": session_id,
            "session_name": session_name,
            "source_type": source_type,
            "start_time": now_iso,
            "status": "active"
        }

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not self.client:
            return self.fallback_repo.get_session(session_id)

        res = self.client.table("sessions").select("*").eq("id", session_id).execute()
        if not res.data:
            return None
        sess = res.data[0]

        # Query predictions count for session
        pred_res = self.client.table("predictions").select("id", count="exact").eq("session_id", session_id).execute()
        total_preds = pred_res.count if pred_res.count is not None else len(pred_res.data)

        started = datetime.fromisoformat(sess["started_at"].replace("Z", "+00:00")) if sess.get("started_at") else None
        ended = datetime.fromisoformat(sess["ended_at"].replace("Z", "+00:00")) if sess.get("ended_at") else None
        if started and started.tzinfo is not None:
            started = started.replace(tzinfo=None)
        if ended and ended.tzinfo is not None:
            ended = ended.replace(tzinfo=None)
        duration = float(sess.get("duration", 0.0))
        if not duration and started and ended:
            duration = round(max((ended - started).total_seconds(), 0.0), 1)

        return {
            "session_id": sess["id"],
            "session_name": sess.get("session_name", "Live Session"),
            "source_type": sess.get("source_type", "webcam"),
            "start_time": sess.get("started_at"),
            "end_time": sess.get("ended_at"),
            "duration_seconds": duration,
            "total_frames": sess.get("total_frames", 0),
            "total_predictions": total_preds,
            "total_people_detected": sess.get("people_count", 0),
            "avg_fps": float(sess.get("avg_fps", 0.0)),
            "dominant_expression": sess.get("dominant_expression", "Neutral"),
            "status": sess.get("status", "completed")
        }

    def end_session(self, session_id: str, total_frames: int = 0, avg_fps: float = 0.0) -> Dict[str, Any]:
        if not self.client:
            return self.fallback_repo.end_session(session_id, total_frames, avg_fps)

        # Get existing session details
        sess_res = self.client.table("sessions").select("*").eq("id", session_id).execute()
        if not sess_res.data:
            return {}

        now_iso = datetime.utcnow().isoformat()
        sess_raw = sess_res.data[0]
        start_iso = sess_raw.get("started_at", now_iso)
        start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00")) if start_iso else datetime.utcnow()
        if start_dt.tzinfo is not None:
            start_dt = start_dt.replace(tzinfo=None)
        now_dt = datetime.utcnow()
        duration = round(max((now_dt - start_dt).total_seconds(), 0.0), 1)

        # Fetch predictions to aggregate stats
        preds_res = self.client.table("predictions").select("person_id, expression, confidence").eq("session_id", session_id).execute()
        preds = preds_res.data or []

        total_preds = len(preds)
        unique_pids = set(p["person_id"] for p in preds if p["person_id"] > 0)
        emotions = [p["expression"] for p in preds if p.get("expression")]
        dominant_emo = Counter(emotions).most_common(1)[0][0] if emotions else "Neutral"

        confidences = [p["confidence"] for p in preds if p.get("confidence") is not None]
        avg_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0

        update_payload = {
            "ended_at": now_iso,
            "duration": duration,
            "people_count": len(unique_pids),
            "total_predictions": total_preds,
            "dominant_expression": dominant_emo,
            "average_confidence": round(avg_conf, 4),
            "status": "completed"
        }
        self.client.table("sessions").update(update_payload).eq("id", session_id).execute()

        return {
            "session_id": session_id,
            "start_time": start_iso,
            "end_time": now_iso,
            "duration_seconds": duration,
            "total_predictions": total_preds,
            "total_people_detected": len(unique_pids),
            "dominant_expression": dominant_emo
        }

    def log_frame_predictions(self, session_id: str, frame_number: int, detections: List[Dict[str, Any]]) -> bool:
        if not self.client:
            return self.fallback_repo.log_frame_predictions(session_id, frame_number, detections)

        try:
            records = []
            now_iso = datetime.utcnow().isoformat()
            for det in detections:
                bbox = det.get("bbox", (0, 0, 0, 0))
                records.append({
                    "session_id": session_id,
                    "person_id": det.get("person_id", -1),
                    "frame_number": frame_number,
                    "timestamp": now_iso,
                    "expression": det.get("emotion", "Neutral"),
                    "confidence": float(det.get("emotion_confidence", 0.0)),
                    "x": int(bbox[0]),
                    "y": int(bbox[1]),
                    "width": int(bbox[2]),
                    "height": int(bbox[3])
                })
            if records:
                def _async_insert(url, key, recs):
                    try:
                        worker_client = create_client(url, key)
                        worker_client.table("predictions").insert(recs).execute()
                    except Exception as err:
                        pass

                self.executor.submit(_async_insert, self.url, self.key, records)
            return True
        except Exception as e:
            print(f"[SupabaseRepository Error] Logging frame predictions failed: {e}")
            return False

    def get_session_analytics(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not self.client:
            return self.fallback_repo.get_session_analytics(session_id)

        sess_res = self.client.table("sessions").select("*").eq("id", session_id).execute()
        if not sess_res.data:
            return None
        sess = sess_res.data[0]

        preds_res = self.client.table("predictions").select("person_id, expression, confidence").eq("session_id", session_id).execute()
        preds = preds_res.data or []

        if not preds:
            return {
                "session_id": session_id,
                "total_people_detected": 0,
                "total_predictions": 0,
                "expression_distribution": {},
                "average_confidence": 0.0,
                "dominant_expression": "None",
                "session_duration_seconds": float(sess.get("duration", 0.0)),
                "avg_fps": float(sess.get("avg_fps", 0.0)),
                "persons": []
            }

        unique_pids = set(p["person_id"] for p in preds if p["person_id"] > 0)
        emotions = [p["expression"] for p in preds if p.get("expression")]
        confidences = [p["confidence"] for p in preds if p.get("confidence") is not None]
        distribution = dict(Counter(emotions))
        dominant_emo = max(distribution, key=distribution.get) if distribution else "Neutral"
        avg_conf = float(sum(confidences) / len(confidences)) * 100 if confidences else 0.0

        return {
            "session_id": session_id,
            "total_people_detected": len(unique_pids),
            "total_predictions": len(preds),
            "expression_distribution": distribution,
            "average_confidence": round(avg_conf, 1),
            "dominant_expression": dominant_emo,
            "session_duration_seconds": float(sess.get("duration", 0.0)),
            "avg_fps": float(sess.get("avg_fps", 0.0)),
            "persons": sorted(list(unique_pids))
        }

    def get_person_analytics(self, session_id: str, person_id: int) -> Optional[Dict[str, Any]]:
        if not self.client:
            return self.fallback_repo.get_person_analytics(session_id, person_id)

        sess_res = self.client.table("sessions").select("id").eq("id", session_id).execute()
        if not sess_res.data:
            return None

        preds_res = self.client.table("predictions")\
                        .select("expression, confidence, frame_number")\
                        .eq("session_id", session_id)\
                        .eq("person_id", person_id)\
                        .order("frame_number", desc=False)\
                        .execute()
        preds = preds_res.data or []
        if not preds:
            return None

        emotions = [p["expression"] for p in preds if p.get("expression")]
        confidences = [p["confidence"] for p in preds if p.get("confidence") is not None]
        distribution = dict(Counter(emotions))
        dominant_emo = max(distribution, key=distribution.get) if distribution else "Neutral"
        avg_conf = float(sum(confidences) / len(confidences)) * 100 if confidences else 0.0

        return {
            "person_id": person_id,
            "dominant_expression": dominant_emo,
            "average_confidence": round(avg_conf, 1),
            "expression_distribution": distribution,
            "expression_timeline": emotions
        }

    def get_latest_frame_detections(self, session_id: str) -> Dict[str, Any]:
        if not self.client:
            return self.fallback_repo.get_latest_frame_detections(session_id)

        sess_res = self.client.table("sessions").select("*").eq("id", session_id).execute()
        if not sess_res.data:
            return {}
        sess = sess_res.data[0]

        preds_res = self.client.table("predictions")\
                        .select("frame_number")\
                        .eq("session_id", session_id)\
                        .order("frame_number", desc=True)\
                        .limit(1)\
                        .execute()
        if not preds_res.data:
            return {
                "session_id": session_id,
                "people_detected": 0,
                "fps": float(sess.get("avg_fps", 0.0)),
                "average_confidence": 0.0,
                "dominant_expression": "None",
                "people": []
            }

        latest_frame_num = preds_res.data[0]["frame_number"]
        frame_preds = self.client.table("predictions")\
                          .select("*")\
                          .eq("session_id", session_id)\
                          .eq("frame_number", latest_frame_num)\
                          .execute().data or []

        people_list = []
        emotions = []
        confidences = []
        for p in frame_preds:
            emotions.append(p.get("expression", "Neutral"))
            confidences.append(float(p.get("confidence", 0.0)))
            people_list.append({
                "person_id": p["person_id"],
                "expression": p.get("expression", "Neutral"),
                "confidence": round(float(p.get("confidence", 0.0)), 2),
                "bounding_box": {
                    "x": p["x"],
                    "y": p["y"],
                    "width": p["width"],
                    "height": p["height"]
                }
            })

        counts = Counter(emotions) if emotions else {}
        dom_emo = counts.most_common(1)[0][0] if counts else "None"
        avg_conf = float(sum(confidences) / len(confidences)) * 100 if confidences else 0.0

        return {
            "session_id": session_id,
            "people_detected": len(people_list),
            "fps": float(sess.get("avg_fps", 30.0)),
            "average_confidence": round(avg_conf, 1),
            "dominant_expression": dom_emo,
            "people": people_list
        }

    def list_sessions(self, page: int = 1, limit: int = 10) -> Dict[str, Any]:
        if not self.client:
            return self.fallback_repo.list_sessions(page, limit)

        count_res = self.client.table("sessions").select("id", count="exact").execute()
        total = count_res.count if count_res.count is not None else len(count_res.data)

        offset = (page - 1) * limit
        sess_res = self.client.table("sessions")\
                       .select("*")\
                       .order("started_at", desc=True)\
                       .range(offset, offset + limit - 1)\
                       .execute()
        sessions_list = []
        for s in sess_res.data or []:
            start_iso = s.get("started_at", "")
            date_str = start_iso[:10] if start_iso else ""
            sessions_list.append({
                "session_id": s["id"],
                "session_name": s.get("session_name") or "Session",
                "date": date_str,
                "duration_seconds": float(s.get("duration") or 0.0),
                "people_count": int(s.get("people_count") or 0),
                "dominant_expression": s.get("dominant_expression") or "Neutral",
                "average_confidence": round(float(s.get("average_confidence") or 0.0) * 100, 1),
                "status": s.get("status") or "completed"
            })

        return {
            "total": total,
            "page": page,
            "limit": limit,
            "sessions": sessions_list
        }


# -------------------------------------------------------------------------
# Factory Function to Instantiate Active Repository Instance
# -------------------------------------------------------------------------
_repository_instance: Optional[DatabaseRepository] = None

def get_db_repository() -> DatabaseRepository:
    """Singleton repository getter based on DATABASE_TYPE environment setting."""
    global _repository_instance
    if _repository_instance is None:
        db_type = settings.DATABASE_TYPE.lower()
        if db_type == "supabase" and settings.SUPABASE_URL and settings.SUPABASE_KEY:
            _repository_instance = SupabaseRepository()
        else:
            if db_type == "supabase":
                print("[DatabaseRepository Warning] SUPABASE_URL or SUPABASE_KEY missing. Falling back to SqliteRepository.")
            _repository_instance = SqliteRepository()
    return _repository_instance
