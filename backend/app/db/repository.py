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

    @abstractmethod
    def save_person_thumbnails(self, session_id: str, persons_details: List[Dict[str, Any]]) -> bool:
        pass


_person_thumbnails_store: Dict[str, List[Dict[str, Any]]] = {}


# -------------------------------------------------------------------------
# Local SQLite Repository Implementation
# -------------------------------------------------------------------------
class SqliteRepository(DatabaseRepository):
    def __init__(self, db_path: str = None):
        path = db_path or str(settings.DATABASE_PATH)
        self.engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)

    def save_person_thumbnails(self, session_id: str, persons_details: List[Dict[str, Any]]) -> bool:
        _person_thumbnails_store[session_id] = persons_details
        return True


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

            persons_details = _person_thumbnails_store.get(session_id, [])
            if not persons_details and unique_pids:
                persons_details = []
                for p_id in sorted(list(unique_pids)):
                    p_logs = [l for l in logs if l.person_id == p_id]
                    p_emos = [l.emotion_label for l in p_logs if l.emotion_label]
                    p_dom = Counter(p_emos).most_common(1)[0][0] if p_emos else "Neutral"
                    p_confs = [l.emotion_confidence for l in p_logs if l.emotion_confidence is not None]
                    p_avg_conf = round(float(sum(p_confs) / len(p_confs)) * 100, 1) if p_confs else 0.0
                    persons_details.append({
                        "person_id": p_id,
                        "thumbnail_b64": "",
                        "dominant_emotion": p_dom,
                        "average_confidence": p_avg_conf,
                        "total_detections": len(p_logs)
                    })

            return {
                "session_id": session_id,
                "total_people_detected": len(unique_pids),
                "total_predictions": len(logs),
                "expression_distribution": distribution,
                "average_confidence": round(avg_conf, 1),
                "dominant_expression": dominant_emo,
                "session_duration_seconds": sess.duration or 0.0,
                "avg_fps": sess.avg_fps or 0.0,
                "persons": sorted(list(unique_pids)),
                "persons_details": persons_details
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
                p_details = _person_thumbnails_store.get(s.session_id, [])
                sessions_list.append({
                    "session_id": s.session_id,
                    "session_name": s.session_name or "Session",
                    "date": s.start_time.strftime("%Y-%m-%d %H:%M") if s.start_time else "",
                    "duration_seconds": s.duration or 0.0,
                    "people_count": s.total_people_detected or 0,
                    "dominant_expression": s.dominant_expression or "Neutral",
                    "average_confidence": round((s.avg_confidence or 0.0) * 100, 1),
                    "status": s.status or "completed",
                    "persons_details": p_details
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

        self.fallback_repo = SqliteRepository()

        if HAS_SUPABASE_SDK and self.url and self.key:
            try:
                self.client = create_client(self.url, self.key)
                print(f"[SupabaseRepository] Connected to Supabase client: {self.url}")
            except Exception as e:
                print(f"[SupabaseRepository Warning] Could not initialize Supabase SDK client: {e}")

    def create_session(self, session_id: str, session_name: str = "Live Session", source_type: str = "webcam") -> Dict[str, Any]:
        local_res = self.fallback_repo.create_session(session_id, session_name, source_type)
        if self.client:
            def _async_create():
                try:
                    now_iso = datetime.utcnow().isoformat()
                    data = {
                        "id": session_id,
                        "session_name": session_name,
                        "source_type": source_type,
                        "started_at": now_iso,
                        "status": "active"
                    }
                    self.client.table("sessions").upsert(data).execute()
                except Exception as err:
                    print(f"[Supabase Sync Warning] create_session sync failed: {err}")
            self.executor.submit(_async_create)
        return local_res

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.fallback_repo.get_session(session_id)

    def end_session(self, session_id: str, total_frames: int = 0, avg_fps: float = 0.0) -> Dict[str, Any]:
        local_res = self.fallback_repo.end_session(session_id, total_frames, avg_fps)
        if self.client:
            def _async_end():
                try:
                    now_iso = datetime.utcnow().isoformat()
                    update_payload = {
                        "ended_at": now_iso,
                        "duration": local_res.get("duration_seconds", 0.0),
                        "people_count": local_res.get("total_people_detected", 0),
                        "total_predictions": local_res.get("total_predictions", 0),
                        "dominant_expression": local_res.get("dominant_expression", "Neutral"),
                        "status": "completed"
                    }
                    self.client.table("sessions").update(update_payload).eq("id", session_id).execute()
                except Exception as err:
                    print(f"[Supabase Sync Warning] end_session sync failed: {err}")
            self.executor.submit(_async_end)
        return local_res

    def log_frame_predictions(self, session_id: str, frame_number: int, detections: List[Dict[str, Any]]) -> bool:
        self.fallback_repo.log_frame_predictions(session_id, frame_number, detections)
        if self.client:
            try:
                records = []
                now_iso = datetime.utcnow().isoformat()
                for det in detections:
                    bbox = det.get("bbox", (0, 0, 0, 0))
                    records.append({
                        "session_id": session_id,
                        "person_id": det.get("person_id", det.get("face_index", -1)),
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
                        except Exception:
                            pass
                    self.executor.submit(_async_insert, self.url, self.key, records)
            except Exception as e:
                print(f"[SupabaseRepository Warning] Logging frame predictions failed: {e}")
        return True

    def get_session_analytics(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.fallback_repo.get_session_analytics(session_id)

    def save_person_thumbnails(self, session_id: str, persons_details: List[Dict[str, Any]]) -> bool:
        self.fallback_repo.save_person_thumbnails(session_id, persons_details)
        if self.client:
            def _async_save_thumbs():
                try:
                    self.client.table("sessions").update({"persons_details": persons_details}).eq("id", session_id).execute()
                except Exception as e:
                    pass
            self.executor.submit(_async_save_thumbs)
        return True

    def get_person_analytics(self, session_id: str, person_id: int) -> Optional[Dict[str, Any]]:
        return self.fallback_repo.get_person_analytics(session_id, person_id)

    def get_latest_frame_detections(self, session_id: str) -> Dict[str, Any]:
        return self.fallback_repo.get_latest_frame_detections(session_id)

    def list_sessions(self, page: int = 1, limit: int = 10) -> Dict[str, Any]:
        return self.fallback_repo.list_sessions(page, limit)


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
