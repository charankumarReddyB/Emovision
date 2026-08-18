"""
Evaluation & Visualization Pipeline for Facial Expression Recognition (FER) Model.
Evaluates EmotionCNN on FER2013 test set, computes sklearn metrics, and generates:
1. Confusion matrix heatmap image (models_eval/confusion_matrix.png)
2. Training vs Validation Accuracy graph (models_eval/accuracy_plot.png)
3. Training vs Validation Loss graph (models_eval/loss_plot.png)
4. Per-class performance breakdown and export to evaluation_report.json
"""
import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Ensure backend path is in sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from app.core.config import settings
from app.ml.model import EmotionCNN
from app.ml.dataset import load_fer2013_data, LABEL_MAP

def run_evaluation_and_generate_plots():
    print("==========================================================================")
    print("       EMOVISION EMOTION RECOGNITION MODEL EVALUATION & PLOTS             ")
    print("==========================================================================")

    models_dir = settings.MODELS_DIR
    model_path = models_dir / "emotion_model.pth"
    history_path = models_dir / "training_history.json"
    
    output_dir = Path(__file__).resolve().parent / "models_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"[ERROR] Trained model weights not found at '{model_path}'.")
        return False

    # 1. Load Dataset
    _, _, test_loader = load_fer2013_data(batch_size=64)
    if test_loader is None:
        print("[ERROR] Test dataset not found.")
        return False

    # Set PyTorch threads for CPU inference
    if not torch.cuda.is_available():
        torch.set_num_threads(os.cpu_count() or 8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading trained weights from '{model_path}' onto {device}...")
    
    model = EmotionCNN(num_classes=7).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    print("Running inference on test dataset (3,589 samples)...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 2. Compute Metrics
    target_names = [LABEL_MAP[i] for i in range(7)]
    test_acc = float(accuracy_score(all_targets, all_preds))
    
    # Macro & Weighted metrics
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    
    # Per-class breakdown
    prec_class, rec_class, f1_class, support_class = precision_recall_fscore_support(all_targets, all_preds, average=None)
    
    cm = confusion_matrix(all_targets, all_preds)

    per_class_metrics = {}
    for i, name in enumerate(target_names):
        per_class_metrics[name] = {
            "precision": float(round(prec_class[i], 4)),
            "recall": float(round(rec_class[i], 4)),
            "f1_score": float(round(f1_class[i], 4)),
            "support": int(support_class[i])
        }

    # Find best and worst performing classes by F1-score
    best_class = max(per_class_metrics.items(), key=lambda x: x[1]["f1_score"])[0]
    worst_class = min(per_class_metrics.items(), key=lambda x: x[1]["f1_score"])[0]

    # Load Training History if available
    train_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    if history_path.exists():
        with open(history_path, "r") as f:
            train_history = json.load(f)

    train_acc = train_history["train_acc"][-1] if train_history.get("train_acc") else 0.0
    val_acc = train_history["val_acc"][-1] if train_history.get("val_acc") else 0.0
    train_loss = train_history["train_loss"][-1] if train_history.get("train_loss") else 0.0
    val_loss = train_history["val_loss"][-1] if train_history.get("val_loss") else 0.0

    print("\n--------------------------------------------------------------------------")
    print(f"Test Accuracy:         {test_acc * 100:.2f}%")
    print(f"Precision (Weighted):  {prec_weighted * 100:.2f}%")
    print(f"Recall (Weighted):     {rec_weighted * 100:.2f}%")
    print(f"F1-Score (Weighted):   {f1_weighted * 100:.2f}%")
    print(f"Best Performing Class: {best_class} (F1: {per_class_metrics[best_class]['f1_score']:.4f})")
    print(f"Worst Performing Class:{worst_class} (F1: {per_class_metrics[worst_class]['f1_score']:.4f})")
    print("--------------------------------------------------------------------------\n")

    # 3. Plot & Save Confusion Matrix Image
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('FER2013 Emotion Recognition — Confusion Matrix')
    plt.xlabel('Predicted Emotion Class')
    plt.ylabel('True Emotion Class')
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Saved] Confusion Matrix plot: {cm_path}")

    # 4. Plot & Save Accuracy Graph
    if train_history.get("train_acc") and train_history.get("val_acc"):
        epochs_range = range(1, len(train_history["train_acc"]) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs_range, [x * 100 for x in train_history["train_acc"]], 'b-o', label='Training Accuracy')
        plt.plot(epochs_range, [x * 100 for x in train_history["val_acc"]], 'g-s', label='Validation Accuracy')
        plt.title('EmotionCNN — Training vs Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='lower right')
        plt.tight_layout()
        acc_path = output_dir / "accuracy_plot.png"
        plt.savefig(acc_path, dpi=300)
        plt.close()
        print(f"[Saved] Accuracy plot: {acc_path}")

    # 5. Plot & Save Loss Graph
    if train_history.get("train_loss") and train_history.get("val_loss"):
        epochs_range = range(1, len(train_history["train_loss"]) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs_range, train_history["train_loss"], 'r-o', label='Training Loss')
        plt.plot(epochs_range, train_history["val_loss"], 'm-s', label='Validation Loss')
        plt.title('EmotionCNN — Training vs Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Cross-Entropy Loss')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='upper right')
        plt.tight_layout()
        loss_path = output_dir / "loss_plot.png"
        plt.savefig(loss_path, dpi=300)
        plt.close()
        print(f"[Saved] Loss plot: {loss_path}")

    # 6. Save JSON Evaluation Report
    report_json = {
        "model": "EmotionCNN (4-block ConvNet)",
        "dataset": "FER2013",
        "num_train_samples": 28709,
        "num_val_samples": 3589,
        "num_test_samples": 3589,
        "test_accuracy": round(test_acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_score_macro": round(f1_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_score_weighted": round(f1_weighted, 4),
        "train_accuracy_final": round(train_acc, 4),
        "val_accuracy_final": round(val_acc, 4),
        "train_loss_final": round(train_loss, 4),
        "val_loss_final": round(val_loss, 4),
        "best_performing_class": best_class,
        "worst_performing_class": worst_class,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist()
    }

    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report_json, f, indent=2)
    print(f"[Saved] Evaluation JSON Report: {report_path}")

    print("\n[SUCCESS] Model Evaluation & Plot Generation Complete!")
    return True

if __name__ == "__main__":
    run_evaluation_and_generate_plots()
