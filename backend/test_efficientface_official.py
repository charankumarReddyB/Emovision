"""
Official Evaluation Suite for Pretrained EfficientFace RAF-DB Model (1.27M Parameters).
Evaluates official EfficientFace model on 3,068 RAF-DB test set images in backend/data/rafdb_test/.

Class Order (EfficientFace ImageFolder):
0 = Neutral
1 = Happy
2 = Sad
3 = Surprise
4 = Fear
5 = Disgust
6 = Angry
"""
import sys
import os
import hashlib
import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Dict, Any, List, Tuple

BACKEND_DIR = Path(__file__).resolve().parent
EFFICIENTFACE_DIR = BACKEND_DIR / "efficientface_repo"

sys.path.insert(0, str(EFFICIENTFACE_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from models.EfficientFace import efficient_face
import cv2

EFFICIENTFACE_CLASSES = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry']

# Map RAF-DB folder label (1..7) to EfficientFace target index
RAF_FOLDER_TO_EFFICIENTFACE_INDEX = {
    1: 3,  # Surprise -> 3
    2: 4,  # Fear -> 4
    3: 5,  # Disgust -> 5
    4: 1,  # Happy -> 1
    5: 2,  # Sad -> 2
    6: 6,  # Angry -> 6
    7: 0,  # Neutral -> 0
}

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

        per_class[EFFICIENTFACE_CLASSES[c]] = {
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

def run_efficientface_evaluation():
    print("=" * 80)
    print("STANDALONE EFFICIENTFACE RAF-DB TEST SET EVALUATION AUDIT")
    print("=" * 80)

    ckpt_path = EFFICIENTFACE_DIR / "checkpoint" / "efficientface_rafdb.pth"
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint missing at {ckpt_path}")
        return

    data_bytes = ckpt_path.read_bytes()
    file_size = len(data_bytes)
    sha256_hash = hashlib.sha256(data_bytes).hexdigest()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load EfficientFace model architecture
    model = efficient_face()
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    
    # Strip module. prefix if saved via DataParallel
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(new_state_dict, strict=True)
    
    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())

    print(f"EFFICIENTFACE CHECKPOINT:\n{ckpt_path}")
    print(f"CHECKPOINT SHA256:\n{sha256_hash}")
    print(f"FILE SIZE:\n{file_size} bytes ({file_size / (1024*1024):.2f} MB)")
    print(f"MODEL PARAMETERS:\n{num_params:,}")
    print(f"FRAMEWORK:\nPyTorch v{torch.__version__}")
    print(f"DEVICE:\n{device}")
    print(f"CLASS MAPPING:\n{dict(enumerate(EFFICIENTFACE_CLASSES))}")

    # Official EfficientFace transforms (PIL RGB image input with specific mean & std)
    normalize = transforms.Normalize(
        mean=[0.57535914, 0.44928582, 0.40079932],
        std=[0.20735591, 0.18981615, 0.18132027]
    )
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    test_dir = BACKEND_DIR / "data" / "rafdb_test" / "DATASET" / "test"
    test_samples: List[Tuple[Path, int]] = []

    if test_dir.exists():
        for class_id in range(1, 8):
            folder = test_dir / str(class_id)
            if folder.exists():
                for img_p in folder.glob("*.*"):
                    if img_p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        eff_idx = RAF_FOLDER_TO_EFFICIENTFACE_INDEX[class_id]
                        test_samples.append((img_p, eff_idx))

    print("\n" + "-" * 80)
    print(f"RAF-DB TEST IMAGES COUNT: {len(test_samples)}")
    print("-" * 80)

    if not test_samples:
        print("ERROR: No RAF-DB test images found in backend/data/rafdb_test/DATASET/test/")
        return

    y_true = []
    y_pred = []

    start_time = time.time()
    batch_size = 64

    with torch.no_grad():
        for idx in range(0, len(test_samples), batch_size):
            batch_samples = test_samples[idx:idx + batch_size]
            tensors = []
            targets = []

            for img_path, target_label in batch_samples:
                pil_img = Image.open(str(img_path)).convert('RGB')
                tensor_img = val_transform(pil_img)
                tensors.append(tensor_img)
                targets.append(target_label)

            batch_tensor = torch.stack(tensors).to(device)
            outputs = model(batch_tensor)
            _, predicts = torch.max(outputs, 1)

            y_true.extend(targets)
            y_pred.extend(predicts.cpu().tolist())

    elapsed = time.time() - start_time
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    res = compute_metrics(y_true_np, y_pred_np)

    print("\n" + "=" * 80)
    print("FINAL EFFICIENTFACE EVALUATION REPORT")
    print("=" * 80)
    print(f"1. EFFICIENTFACE CHECKPOINT PATH : {ckpt_path}")
    print(f"2. CHECKPOINT SHA256            : {sha256_hash}")
    print(f"3. PARAMETER COUNT              : {num_params:,}")
    print(f"4. INPUT RESOLUTION             : 3x224x224 RGB PIL Format")
    print(f"5. RAF-DB TEST IMAGE COUNT      : {len(y_true_np)}")
    print(f"6. ACTUAL MEASURED ACCURACY     : {res['accuracy']}%")
    print(f"7. MACRO PRECISION              : {res['macro_precision']}%")
    print(f"8. MACRO RECALL                 : {res['macro_recall']}%")
    print(f"9. MACRO F1                     : {res['macro_f1']}%")
    print(f"10. WEIGHTED F1                 : {res['weighted_f1']}%")
    print(f"EVALUATION TIME                 : {elapsed:.2f} seconds ({len(y_true_np)/elapsed:.1f} img/sec)")

    print("\n11. PER-CLASS RESULTS:")
    for label_name, p_dict in res['per_class'].items():
        print(f"    {label_name:10s} -> F1: {p_dict['f1']:6.2f}% | Precision: {p_dict['precision']:6.2f}% | Recall: {p_dict['recall']:6.2f}% (Support: {p_dict['support']})")

    print("\n12. CONFUSION MATRIX (Rows: Ground Truth, Cols: Prediction):")
    print(f"    {'':10s}" + "".join([f"{l[:7]:>9s}" for l in EFFICIENTFACE_CLASSES]))
    for c_idx, row in enumerate(res['confusion_matrix']):
        row_str = "".join([f"{val:9d}" for val in row])
        print(f"    {EFFICIENTFACE_CLASSES[c_idx]:10s}{row_str}")

    passed_80 = res['accuracy'] >= 80.0
    print(f"\n13. PASSED 80% REQUIREMENT      : {'YES' if passed_80 else 'NO'}")
    print("=" * 80)

if __name__ == "__main__":
    run_efficientface_evaluation()
