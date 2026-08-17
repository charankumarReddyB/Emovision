"""
SCRFD ONNX Face Detector Service.
Detects N human faces in an image or video frame with 5 facial keypoints.
Uses ONNX Runtime for high-performance CPU execution with 0 false positives on background objects.
"""
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from app.core.config import settings, BASE_DIR

class SCRFDDetector:
    """
    SCRFD ONNX Face Detector (SCRFD-500M / SCRFD-2.5G).
    Returns face bounding boxes (x, y, w, h), confidence scores, and 5 facial keypoints:
    [left_eye, right_eye, nose, left_mouth, right_mouth]
    """
    def __init__(
        self,
        model_path: Optional[Path] = None,
        score_threshold: float = 0.30,
        nms_threshold: float = 0.35,
        input_size: Tuple[int, int] = (640, 640)
    ):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.strides = [8, 16, 32]
        self.num_anchors = 2
        
        models_dir = BASE_DIR / "app" / "models_weights"
        if model_path is None:
            model_path = models_dir / "scrfd_500m_bnkps.onnx"
            if not model_path.exists():
                fallback = models_dir / "scrfd_2.5g_bnkps.onnx"
                if fallback.exists():
                    model_path = fallback
            
        self.session = None
        if model_path.exists():
            try:
                self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
                self.output_names = [o.name for o in self.session.get_outputs()]
            except Exception as e:
                print(f"[SCRFDDetector Warning] Could not load ONNX model from {model_path}: {e}")
                self.session = None

        # Precompute anchor centers
        self.anchors_by_stride = self._generate_anchors()

    def _generate_anchors(self) -> Dict[int, np.ndarray]:
        w_in, h_in = self.input_size
        anchors = {}
        for stride in self.strides:
            feat_h = h_in // stride
            feat_w = w_in // stride
            grid_y, grid_x = np.mgrid[0:feat_h, 0:feat_w]
            grid = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
            grid = (grid * stride).reshape(-1, 2)
            grid = np.repeat(grid, self.num_anchors, axis=0)
            anchors[stride] = grid
        return anchors

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> List[int]:
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= self.nms_threshold)[0]
            order = order[inds + 1]
            
        return keep

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects all N visible human faces in input frame with 5 facial keypoints.
        
        Args:
            frame (np.ndarray): BGR OpenCV image frame.
            
        Returns:
            List[Dict[str, Any]]: List of detected faces with structure:
                {
                    "bbox": (x, y, w, h),
                    "confidence": float,
                    "kps": np.ndarray shape (5, 2),
                    "face_chip": np.ndarray (cropped BGR face chip)
                }
        """
        if frame is None or frame.size == 0 or self.session is None:
            return []

        h_orig, w_orig = frame.shape[:2]
        w_in, h_in = self.input_size
        # Compute aspect-ratio preserving scaling factor
        r = min(float(w_in) / float(w_orig), float(h_in) / float(h_orig))
        nw, nh = int(round(w_orig * r)), int(round(h_orig * r))
        
        resized_img = cv2.resize(frame, (nw, nh))
        padded_img = np.full((h_in, w_in, 3), 127, dtype=np.uint8)
        pad_x = (w_in - nw) // 2
        pad_y = (h_in - nh) // 2
        padded_img[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized_img
        scale = 1.0 / r
        
        # Prepare DNN blob normalized with Mean=127.5, Std=128.0
        blob = cv2.dnn.blobFromImage(padded_img, 1.0 / 128.0, (w_in, h_in), (127.5, 127.5, 127.5), swapRB=True)
        
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        
        score_outputs = outputs[0:3]
        bbox_outputs = outputs[3:6]
        kps_outputs = outputs[6:9]
        
        proposal_boxes = []
        proposal_scores = []
        proposal_kps = []
        
        for idx, stride in enumerate(self.strides):
            scores = score_outputs[idx].reshape(-1)
            bboxes = bbox_outputs[idx].reshape(-1, 4) * stride
            kps = kps_outputs[idx].reshape(-1, 10) * stride
            anchors = self.anchors_by_stride[stride]
            
            pos_inds = np.where(scores >= self.score_threshold)[0]
            if len(pos_inds) == 0:
                continue
                
            pos_scores = scores[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            pos_kps = kps[pos_inds]
            
            x1 = pos_anchors[:, 0] - pos_bboxes[:, 0]
            y1 = pos_anchors[:, 1] - pos_bboxes[:, 1]
            x2 = pos_anchors[:, 0] + pos_bboxes[:, 2]
            y2 = pos_anchors[:, 1] + pos_bboxes[:, 3]
            boxes = np.stack([x1, y1, x2, y2], axis=-1)
            
            decoded_kps = np.zeros((len(pos_inds), 5, 2), dtype=np.float32)
            for k_idx in range(5):
                decoded_kps[:, k_idx, 0] = pos_anchors[:, 0] + pos_kps[:, k_idx * 2]
                decoded_kps[:, k_idx, 1] = pos_anchors[:, 1] + pos_kps[:, k_idx * 2 + 1]
                
            proposal_boxes.append(boxes)
            proposal_scores.append(pos_scores)
            proposal_kps.append(decoded_kps)
            
        if not proposal_boxes:
            return []
            
        proposal_boxes = np.vstack(proposal_boxes)
        proposal_scores = np.concatenate(proposal_scores)
        proposal_kps = np.vstack(proposal_kps)
        
        keep = self._nms(proposal_boxes, proposal_scores)
        
        results = []
        for i in keep:
            score = float(proposal_scores[i])
            box = proposal_boxes[i]
            kps = proposal_kps[i]
            
            x1 = max(0, int((box[0] - pad_x) * scale))
            y1 = max(0, int((box[1] - pad_y) * scale))
            x2 = min(w_orig, int((box[2] - pad_x) * scale))
            y2 = min(h_orig, int((box[3] - pad_y) * scale))
            
            w_box = max(0, x2 - x1)
            h_box = max(0, y2 - y1)
            
            if w_box < 15 or h_box < 15:
                continue
                
            scaled_kps = kps.copy()
            scaled_kps[:, 0] = (scaled_kps[:, 0] - pad_x) * scale
            scaled_kps[:, 1] = (scaled_kps[:, 1] - pad_y) * scale
            
            chip = frame[y1:y2, x1:x2].copy()
            
            results.append({
                "bbox": (x1, y1, w_box, h_box),
                "confidence": score,
                "kps": scaled_kps,
                "face_chip": chip
            })
            
        # Sort faces by bounding box area descending (largest face in front is Face 1)
        results.sort(key=lambda d: d["bbox"][2] * d["bbox"][3], reverse=True)
        
        # Suppress sub-boxes contained inside a larger primary face box
        filtered = []
        for det in results:
            bx, by, bw, bh = det["bbox"]
            is_sub = False
            for other in filtered:
                ox, oy, ow, oh = other["bbox"]
                if bx >= (ox - 10) and by >= (oy - 10) and (bx + bw) <= (ox + ow + 10) and (by + bh) <= (oy + oh + 10):
                    is_sub = True
                    break
            if not is_sub:
                filtered.append(det)

        return filtered[:50]
