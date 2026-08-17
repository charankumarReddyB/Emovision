"""
Standalone Verification & Audit Suite for PyTorch DAN Model on RAF-DB Test Set.
Evaluates official DAN model (ResNet-18 + 4 Cross-Attention Heads) on RAF-DB Test Set.

RAF-DB Ground Truth Folders / CSV Labels:
1 = Surprise -> DAN Index 0
2 = Fear     -> DAN Index 1
3 = Disgust  -> DAN Index 2
4 = Happy    -> DAN Index 3
5 = Sad      -> DAN Index 4
6 = Angry    -> DAN Index 5
7 = Neutral  -> DAN Index 6
"""
import sys
import os
import csv
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from pathlib import Path
from typing import List, Dict, Tuple

# Add backend directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.core.config import BASE_DIR
from app.ml.dan import DAN, DAN_RAFDB_LABELS
from app.services.face_aligner import FaceAligner

# Map RAF-DB folder label (1..7) to DAN model output index (0..6)
RAF_LABEL_TO_DAN_INDEX = {
    1: 0,  # Surprise
    2: 1,  # Fear
    3: 2,  # Disgust
    4: 3,  # Happy
    5: 4,  # Sad
    6: 5,  # Angry
    7: 6,  # Neutral
}

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Computes Accuracy, Macro Precision/Recall/F1, Weighted F1, Per-class metrics, and Confusion Matrix."""
    num_classes = 7
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    total = len(y_true)
    correct = np.trace(cm)
    accuracy = (correct / total * 100.0) if total > 0 else 0.0

    precisions = []
    recalls = []
    f1s = []
    support = []

    per_class = {}
    for c in range(num_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        sup = np.sum(cm[c, :])

        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        support.append(sup)

        per_class[DAN_RAFDB_LABELS[c]] = {
            "precision": round(prec * 100.0, 2),
            "recall": round(rec * 100.0, 2),
            "f1": round(f1 * 100.0, 2),
            "support": int(sup)
        }

    macro_precision = np.mean(precisions) * 100.0
    macro_recall = np.mean(recalls) * 100.0
    macro_f1 = np.mean(f1s) * 100.0
    weighted_f1 = (np.sum(np.array(f1s) * np.array(support)) / total * 100.0) if total > 0 else 0.0

    return {
        "accuracy": round(accuracy, 2),
        "macro_precision": round(macro_precision, 2),
        "macro_recall": round(macro_recall, 2),
        "macro_f1": round(macro_f1, 2),
        "weighted_f1": round(weighted_f1, 2),
        "per_class": per_class,
        "confusion_matrix": cm
    }

def verify_and_evaluate():
    print("=" * 80)
    print("STANDALONE DAN MODEL RAF-DB TEST SET EVALUATION AUDIT")
    print("=" * 80)

    checkpoint_path = BASE_DIR / "app" / "models_weights" / "dan_rafdb.pth"
    
    if not checkpoint_path.exists():
        print(f"DAN MODEL:\nNot Found at {checkpoint_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DAN(num_class=7, pretrained=False).to(device)

    data_bytes = checkpoint_path.read_bytes()
    sha256_hash = hashlib.sha256(data_bytes).hexdigest()

    ckpt_dict = torch.load(str(checkpoint_path), map_location=device)
    state_dict = ckpt_dict['model_state_dict'] if isinstance(ckpt_dict, dict) and 'model_state_dict' in ckpt_dict else (ckpt_dict if isinstance(ckpt_dict, dict) else ckpt_dict)
    
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"DAN MODEL:\nLoaded ({checkpoint_path.name})")
    print(f"PATH:\n{checkpoint_path}")
    print(f"FILE SIZE:\n{len(data_bytes)} bytes ({len(data_bytes)/(1024*1024):.2f} MB)")
    print(f"SHA256 HASH:\n{sha256_hash}")
    print(f"STRICT LOADING:\nPassed (0 missing, 0 unexpected)")
    print(f"DEVICE:\n{device}")
    print(f"CLASS ORDER:\n{dict(enumerate(DAN_RAFDB_LABELS))}")

    aligner = FaceAligner(target_size=(224, 224))

    # Look for test dataset in backend/data/rafdb_test/
    raf_test_dir = BASE_DIR / "data" / "rafdb_test"
    dataset_dir = raf_test_dir / "DATASET" / "test"
    csv_labels_path = raf_test_dir / "test_labels.csv"

    # Collect images to evaluate
    test_samples: List[Tuple[Path, int]] = []

    if dataset_dir.exists():
        for class_folder in range(1, 8):
            folder_path = dataset_dir / str(class_folder)
            if folder_path.exists():
                for img_file in folder_path.glob("*.*"):
                    if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        dan_idx = RAF_LABEL_TO_DAN_INDEX[class_folder]
                        test_samples.append((img_file, dan_idx))

    if not test_samples and csv_labels_path.exists():
        with open(csv_labels_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    img_name, raf_label = row[0].strip(), int(row[1].strip())
                    img_path = raf_test_dir / img_name
                    if img_path.exists() and raf_label in RAF_LABEL_TO_DAN_INDEX:
                        test_samples.append((img_path, RAF_LABEL_TO_DAN_INDEX[raf_label]))

    if not test_samples:
        print("\n" + "=" * 80)
        print("RAF-DB TEST DATASET STATUS")
        print("=" * 80)
        print("RAF-DB test set must be copied from Google Drive/Colab into backend/data/rafdb_test.")
        print("Expected Structure:")
        print("backend/data/rafdb_test/\n|-- test_labels.csv\n+-- DATASET/\n    +-- test/\n        |-- 1/\n        |-- 2/\n        |-- 3/\n        |-- 4/\n        |-- 5/\n        |-- 6/\n        +-- 7/")
        print("=" * 80)
        return

    print("\n" + "-" * 80)
    print(f"EVALUATING {len(test_samples)} RAF-DB TEST IMAGES ON DAN MODEL...")
    print("-" * 80)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for img_path, true_dan_idx in test_samples:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Standard DAN preprocessing (224x224 RGB ImageNet normalized CHW tensor)
            preprocessed_chw = aligner.preprocess_aligned_face_pytorch(img)
            input_tensor = torch.from_numpy(preprocessed_chw).unsqueeze(0).to(device)

            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)[0]
            pred_dan_idx = int(torch.argmax(probs))

            y_true.append(true_dan_idx)
            y_pred.append(pred_dan_idx)

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    res = compute_metrics(y_true_np, y_pred_np)

    print("\n" + "=" * 80)
    print("FINAL EVALUATION METRICS REPORT")
    print("=" * 80)
    print(f"DAN MODEL:\nLoaded")
    print(f"\nRAF-DB TEST IMAGES:\n{len(y_true_np)}")
    print(f"\nTEST ACCURACY:\n{res['accuracy']}%")
    print(f"\nMACRO PRECISION:\n{res['macro_precision']}%")
    print(f"\nMACRO RECALL:\n{res['macro_recall']}%")
    print(f"\nMACRO F1:\n{res['macro_f1']}%")
    print(f"\nWEIGHTED F1:\n{res['weighted_f1']}%")

    print("\nPER CLASS:")
    for label_name, p_dict in res['per_class'].items():
        print(f"{label_name:10s} -> Precision: {p_dict['precision']:6.2f}% | Recall: {p_dict['recall']:6.2f}% | F1: {p_dict['f1']:6.2f}% (Support: {p_dict['support']})")

    print("\nCONFUSION MATRIX (Rows: Ground Truth, Cols: Prediction):")
    print(f"{'':10s}" + "".join([f"{l[:7]:>9s}" for l in DAN_RAFDB_LABELS]))
    for idx, row in enumerate(res['confusion_matrix']):
        row_str = "".join([f"{val:9d}" for val in row])
        print(f"{DAN_RAFDB_LABELS[idx]:10s}{row_str}")

    print("\n" + "=" * 80)
    if res['accuracy'] >= 80.0:
        print("[CONCLUSION] MODEL ACCURACY >= 80% — READY FOR APPLICATION INTEGRATION!")
    else:
        print("[CONCLUSION] MODEL ACCURACY < 80% — NOT READY FOR APPLICATION INTEGRATION.")
        print("Investigate preprocessing / checkpoint / label mapping alignment.")
    print("=" * 80)

if __name__ == "__main__":
    verify_and_evaluate()
