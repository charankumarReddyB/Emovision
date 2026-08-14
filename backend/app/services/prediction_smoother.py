"""
Temporal Prediction Smoothing Module for Facial Expression Recognition.
Applies sliding-window majority voting and weighted confidence averaging per Person ID
to prevent rapid frame-to-frame expression flickering.
"""
from collections import deque, Counter
from typing import Dict, List, Tuple, Any
from app.core.config import settings

class PredictionSmoother:
    """
    Smooths raw emotion predictions across consecutive video frames per tracked Person ID.
    """
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        # person_id -> deque of (emotion_label, confidence)
        self.history: Dict[int, deque] = {}

    def smooth_prediction(self, person_id: int, raw_emotion: str, raw_confidence: float) -> Tuple[str, float]:
        """
        Adds raw prediction to history queue for person_id and returns smoothed (emotion, confidence).
        
        Args:
            person_id (int): Tracked persistent Person ID.
            raw_emotion (str): Raw model prediction for current frame.
            raw_confidence (float): Raw model confidence for current frame.
            
        Returns:
            Tuple[str, float]: (Smoothed Emotion Label, Smoothed Confidence Score)
        """
        if person_id not in self.history:
            self.history[person_id] = deque(maxlen=self.window_size)
            
        self.history[person_id].append((raw_emotion, raw_confidence))
        
        # Extract recent emotion labels
        recent_emotions = [item[0] for item in self.history[person_id]]
        
        # Majority Vote
        counts = Counter(recent_emotions)
        majority_emotion, _ = counts.most_common(1)[0]
        
        # Average confidence for the majority emotion in window
        majority_confidences = [item[1] for item in self.history[person_id] if item[0] == majority_emotion]
        avg_confidence = float(sum(majority_confidences) / len(majority_confidences)) if majority_confidences else raw_confidence
        
        return majority_emotion, round(avg_confidence, 4)

    def cleanup_inactive_tracks(self, active_person_ids: List[int]):
        """Removes history for Person IDs that are no longer active."""
        active_set = set(active_person_ids)
        for pid in list(self.history.keys()):
            if pid not in active_set:
                del self.history[pid]
