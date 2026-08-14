"""
Evaluation Pipeline for Facial Expression Recognition (FER) Model.
Calculates Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and Per-Class Breakdown.
Saves metrics to backend/data/evaluation_metrics.json for frontend analytics dashboard.
"""
import torch
import numpy as np
import json
from pathlib import Path
import sys
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Add backend directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.core.config import settings
from app.ml.model import EmotionCNN
from app.ml.dataset import load_fer2013_data, LABEL_MAP

def evaluate_model():
    """
    Evaluates trained EmotionCNN on FER2013 test set.
    """
    print("\n=======================================================")
    print("EMOVISION FACIAL EXPRESSION RECOGNITION - EVALUATION")
    print("=======================================================")

    model_path = settings.MODELS_DIR / "emotion_model.pth"
    output_metrics_path = settings.DATA_DIR / "evaluation_metrics.json"

    # Check if trained weights exist
    if not model_path.exists():
        print(f"\n[MODEL WEIGHTS MISSING ERROR]")
        print(f"No trained model weights found at: '{model_path}'")
        print(f"Please train the model first using: python backend/app/ml/train.py")
        print(f"Evaluation safely halted without hardcoding fake metrics.")
        return False

    # Check if dataset exists
    _, _, test_loader = load_fer2013_data()
    if test_loader is None:
        print("\n[DATASET MISSING ERROR] Cannot run evaluation without FER2013 dataset.")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN(num_classes=7).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    print(f"Evaluating model on test dataset...")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute Metrics using scikit-learn
    acc = accuracy_score(all_targets, all_preds)
    target_names = [LABEL_MAP[i] for i in range(7)]
    report_dict = classification_report(all_targets, all_preds, target_names=target_names, output_dict=True)
    conf_matrix = confusion_matrix(all_targets, all_preds).tolist()

    print(f"\n---------------- EVALUATION METRICS RESULTS ----------------")
    print(f"Overall Accuracy: {acc*100:.2f}%")
    print(f"Macro F1-Score:   {report_dict['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1-Score: {report_dict['weighted avg']['f1-score']:.4f}")
    print(f"----------------------------------------------------------")

    # Format Per-Class breakdown
    per_class_metrics = {}
    for i, name in enumerate(target_names):
        if name in report_dict:
            per_class_metrics[name] = {
                "precision": round(report_dict[name]["precision"], 4),
                "recall": round(report_dict[name]["recall"], 4),
                "f1_score": round(report_dict[name]["f1-score"], 4),
                "support": report_dict[name]["support"]
            }
            print(f"{name:10s} | Precision: {report_dict[name]['precision']:.4f} | Recall: {report_dict[name]['recall']:.4f} | F1: {report_dict[name]['f1-score']:.4f}")

    # Export Evaluation Results to JSON for Dashboard
    metrics_payload = {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(report_dict['macro avg']['f1-score']), 4),
        "weighted_f1": round(float(report_dict['weighted avg']['f1-score']), 4),
        "confusion_matrix": conf_matrix,
        "per_class_performance": per_class_metrics,
        "target_emotions": target_names
    }

    with open(output_metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\nEvaluation metrics successfully exported to '{output_metrics_path}'")
    return True

if __name__ == "__main__":
    evaluate_model()
