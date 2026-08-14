"""
Multi-Person Face Tracking Module.
Uses Centroid Euclidean Distance and Bounding Box IoU (Intersection over Union) matching
to assign and maintain unique persistent Person IDs across video/webcam frames.
"""
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from app.core.config import settings

def calculate_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes (x, y, w, h).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0

    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    
    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return float(iou)

def get_centroid(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Computes the (cx, cy) centroid of a bounding box (x, y, w, h)."""
    return (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)

class FaceTracker:
    """
    Tracks multiple faces across frames and assigns persistent unique Person IDs.
    """
    def __init__(
        self,
        max_disappeared: int = settings.MAX_DISAPPEARED_FRAMES,
        max_distance: float = settings.MAX_CENTROID_DISTANCE,
        iou_threshold: float = settings.IOU_THRESHOLD
    ):
        self.next_person_id: int = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}  # person_id -> track info
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.iou_threshold = iou_threshold

    def register(self, bbox: Tuple[int, int, int, int], face_chip: Optional[np.ndarray] = None) -> int:
        """Registers a new face track with a unique Person ID."""
        person_id = self.next_person_id
        centroid = get_centroid(bbox)
        self.tracks[person_id] = {
            "bbox": bbox,
            "centroid": centroid,
            "disappeared": 0,
            "face_chip": face_chip,
            "total_frames_seen": 1
        }
        self.next_person_id += 1
        return person_id

    def unregister(self, person_id: int):
        """Removes a track when a person is lost/leaves the frame."""
        if person_id in self.tracks:
            del self.tracks[person_id]

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Updates tracked Person IDs with current frame face detections.
        
        Args:
            detections (List[Dict[str, Any]]): Face detections from FaceDetector.
            
        Returns:
            List[Dict[str, Any]]: Updated detections with assigned 'person_id'.
        """
        if len(detections) == 0:
            # Mark all existing tracks as disappeared
            for person_id in list(self.tracks.keys()):
                self.tracks[person_id]["disappeared"] += 1
                if self.tracks[person_id]["disappeared"] > self.max_disappeared:
                    self.unregister(person_id)
            return []

        # Extract input centroids & bboxes
        input_bboxes = [d["bbox"] for d in detections]
        input_centroids = np.array([get_centroid(b) for b in input_bboxes])

        # If currently tracking no faces, register all incoming detections
        if len(self.tracks) == 0:
            for i, det in enumerate(detections):
                pid = self.register(input_bboxes[i], det.get("face_chip"))
                det["person_id"] = pid
            return detections

        # Grab existing track IDs and centroids
        track_ids = list(self.tracks.keys())
        track_centroids = np.array([self.tracks[pid]["centroid"] for pid in track_ids])
        track_bboxes = [self.tracks[pid]["bbox"] for pid in track_ids]

        # Compute Euclidean distance matrix between track centroids and detection centroids
        D = np.linalg.norm(track_centroids[:, np.newaxis] - input_centroids, axis=2)

        # Find smallest distance matches
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            # Verify distance and IoU constraints
            dist = D[row, col]
            iou = calculate_iou(track_bboxes[row], input_bboxes[col])
            
            # Match if within max spatial distance or above IoU threshold
            if dist <= self.max_distance or iou >= self.iou_threshold:
                pid = track_ids[row]
                self.tracks[pid]["bbox"] = input_bboxes[col]
                self.tracks[pid]["centroid"] = input_centroids[col]
                self.tracks[pid]["disappeared"] = 0
                self.tracks[pid]["total_frames_seen"] += 1
                self.tracks[pid]["face_chip"] = detections[col].get("face_chip")
                
                detections[col]["person_id"] = pid

                used_rows.add(row)
                used_cols.add(col)

        # Handle unmatched existing tracks
        unused_rows = set(range(0, D.shape[0])) - used_rows
        for row in unused_rows:
            pid = track_ids[row]
            self.tracks[pid]["disappeared"] += 1
            if self.tracks[pid]["disappeared"] > self.max_disappeared:
                self.unregister(pid)

        # Handle unmatched new detections (new people entering frame)
        unused_cols = set(range(0, D.shape[1])) - used_cols
        for col in unused_cols:
            pid = self.register(input_bboxes[col], detections[col].get("face_chip"))
            detections[col]["person_id"] = pid

        return detections
