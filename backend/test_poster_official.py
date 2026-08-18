"""
Official Evaluation Suite for Pretrained POSTER RAF-DB Model (71.85M Parameters).
Evaluates official POSTER model on our 3,068 RAF-DB test set images in backend/data/rafdb_test/.

Label Mapping (POSTER / RAF-DB Standard):
0 = Surprise (SU)
1 = Fear (FE)
2 = Disgust (DI)
3 = Happy (HA)
4 = Sad (SA)
5 = Angry (AN)
6 = Neutral (NE)
"""
import sys
import os
import hashlib
import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
from typing import Dict, Any

# Add poster_repo and backend to Python path
BACKEND_DIR = Path(__file__).resolve().parent
POSTER_DIR = BACKEND_DIR / "poster_repo"

sys.path.insert(0, str(POSTER_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from models.emotion_hyp import pyramid_trans_expr
from utils import load_pretrained_weights
import cv2

DAN_POSTER_LABELS = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']
RAF_LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
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

        per_class[DAN_POSTER_LABELS[c]] = {
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

def run_poster_evaluation():
    print("=" * 80)
    print("OFFICIAL PRETRAINED POSTER RAF-DB EVALUATION SUITE")
    print("=" * 80)

    # 1. Checkpoint Verification
    ckpt_path = POSTER_DIR / "checkpoint" / "rafdb_best.pth"
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint missing at {ckpt_path}")
        return

    data_bytes = ckpt_path.read_bytes()
    file_size = len(data_bytes)
    sha256_hash = hashlib.sha256(data_bytes).hexdigest()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Change working directory to poster_repo for relative pretrain backbone loading
    orig_cwd = os.getcwd()
    os.chdir(str(POSTER_DIR))

    try:
        model = pyramid_trans_expr(img_size=224, num_classes=7, type='large')
        checkpoint = torch.load('checkpoint/rafdb_best.pth', map_location=device)
        model_state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model = load_pretrained_weights(model, model_state)
        model = model.to(device)
        model.eval()
    finally:
        os.chdir(orig_cwd)

    num_params = sum(p.numel() for p in model.parameters())

    print(f"POSTER CHECKPOINT:\n{ckpt_path}")
    print(f"CHECKPOINT SHA256:\n{sha256_hash}")
    print(f"FILE SIZE:\n{file_size} bytes ({file_size / (1024*1024):.2f} MB)")
    print(f"MODEL PARAMETERS:\n{num_params:,}")
    print(f"FRAMEWORK:\nPyTorch v{torch.__version__}")
    print(f"DEVICE:\n{device}")
    print(f"INPUT RESOLUTION:\n[3, 224, 224] BGR format with ImageNet Normalization")
    print(f"PREPROCESSING:\nResize(224,224) -> ToTensor() -> Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])")
    print(f"CLASS MAPPING:\n{dict(enumerate(DAN_POSTER_LABELS))}")

    # 3. Collect 3,068 RAF-DB test images
    test_dir = BACKEND_DIR / "data" / "rafdb_test" / "DATASET" / "test"
    test_samples = []

    if test_dir.exists():
        for class_id in range(1, 8):
            folder = test_dir / str(class_id)
            if folder.exists():
                for img_p in folder.glob("*.*"):
                    if img_p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        test_samples.append((img_p, RAF_LABEL_MAP[class_id]))

    print("\n" + "-" * 80)
    print(f"RAF-DB TEST IMAGES COUNT: {len(test_samples)}")
    print("-" * 80)

    if not test_samples:
        print("ERROR: No RAF-DB test images found in backend/data/rafdb_test/DATASET/test/")
        return

    # Official POSTER preprocessing pipeline (BGR format as expected by POSTER OpenCV reader)
    data_transforms_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    y_true = []
    y_pred = []

    start_time = time.time()
    batch_size = 32

    with torch.no_grad():
        for idx in range(0, len(test_samples), batch_size):
            batch_samples = test_samples[idx:idx + batch_size]
            tensors = []
            targets = []

            for img_path, target_label in batch_samples:
                # Read using cv2 (BGR)
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                tensor_img = data_transforms_test(img)
                tensors.append(tensor_img)
                targets.append(target_label)

            if not tensors:
                continue

            batch_tensor = torch.stack(tensors).to(device)
            outputs, _ = model(batch_tensor)
            _, predicts = torch.max(outputs, 1)

            y_true.extend(targets)
            y_pred.extend(predicts.cpu().tolist())

    elapsed = time.time() - start_time
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    res = compute_metrics(y_true_np, y_pred_np)

    print("\n" + "=" * 80)
    print("FINAL POSTER EVALUATION REPORT")
    print("=" * 80)
    print(f"1. POSTER CHECKPOINT PATH     : {ckpt_path}")
    print(f"2. CHECKPOINT SHA256           : {sha256_hash}")
    print(f"3. PARAMETER COUNT             : {num_params:,}")
    print(f"4. INPUT RESOLUTION            : 3x224x224")
    print(f"5. PREPROCESSING               : OpenCV BGR -> Resize(224,224) -> ToTensor() -> ImageNet Norm")
    print(f"6. RAF-DB TEST IMAGE COUNT     : {len(y_true_np)}")
    print(f"7. ACTUAL ACCURACY             : {res['accuracy']}%")
    print(f"8. MACRO F1                    : {res['macro_f1']}%")
    print(f"9. WEIGHTED F1                 : {res['weighted_f1']}%")
    print(f"EVALUATION TIME                : {elapsed:.2f} seconds ({len(y_true_np)/elapsed:.1f} img/sec)")

    print("\n10. PER-CLASS RESULTS:")
    for label_name, p_dict in res['per_class'].items():
        print(f"    {label_name:10s} -> F1: {p_dict['f1']:6.2f}% | Precision: {p_dict['precision']:6.2f}% | Recall: {p_dict['recall']:6.2f}% (Support: {p_dict['support']})")

    print("\n11. CONFUSION MATRIX (Rows: Ground Truth, Cols: Prediction):")
    print(f"    {'':10s}" + "".join([f"{l[:7]:>9s}" for l in DAN_POSTER_LABELS]))
    for c_idx, row in enumerate(res['confusion_matrix']):
        row_str = "".join([f"{val:9d}" for val in row])
        print(f"    {DAN_POSTER_LABELS[c_idx]:10s}{row_str}")

    passed_80 = res['accuracy'] >= 80.0
    print(f"\n12. PASSED 80% REQUIREMENT     : {'YES' if passed_80 else 'NO'}")
    print(f"13. INTEGRATION WITH SCRFD     : {'STARTED' if passed_80 else 'HALTED'}")
    print("=" * 80)

if __name__ == "__main__":
    run_poster_evaluation()
