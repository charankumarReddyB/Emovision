"""
Experiment 1: Class-Weighted Loss Training Pipeline for Facial Expression Recognition.
Calculates dynamic inverse-frequency class weights from training set distribution and trains
EmotionCNN using weighted CrossEntropyLoss. Evaluates on untouched PrivateTest set (3,589 samples).
"""
import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

sys.path.append(str(Path(__file__).resolve().parent))

from app.core.config import settings
from app.ml.model import EmotionCNN
from app.ml.dataset import load_fer2013_data, LABEL_MAP

def run_experiment_1():
    print("==========================================================================")
    print("          EXPERIMENT 1: CLASS-WEIGHTED LOSS TRAINING PIPELINE             ")
    print("==========================================================================")

    # 1. Load Dataset
    batch_size = 128
    train_loader, val_loader, test_loader = load_fer2013_data(batch_size=batch_size)
    if train_loader is None or test_loader is None:
        print("[ERROR] FER2013 dataset not found!")
        return False

    # 2. Calculate Dynamic Class Weights from Training Set
    csv_path = settings.DATA_DIR / "dataset" / "fer2013" / "fer2013.csv"
    df = pd.read_csv(csv_path)
    train_df = df[df['Usage'] == 'Training'] if 'Usage' in df.columns else df
    
    total_train_samples = len(train_df)
    num_classes = 7
    class_counts = train_df['emotion'].value_counts().sort_index().to_dict()
    
    weights = []
    print("\nDynamic Class Weight Calculation (Inverse Frequency):")
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        # w_c = N_total / (N_classes * N_c)
        w = total_train_samples / (num_classes * count)
        weights.append(w)
        print(f"  Class {i} ({LABEL_MAP[i]:10s}): Count={count:5d} | Weight={w:.4f}")

    if not torch.cuda.is_available():
        torch.set_num_threads(os.cpu_count() or 8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"\nUsing compute device: {device} (CPU Threads: {torch.get_num_threads()})")

    # 3. Instantiate Model, Weighted Loss, Optimizer, Scheduler
    model = EmotionCNN(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    epochs = 20
    patience = 7
    patience_counter = 0
    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\nStarting Experiment 1 Training ({epochs} Max Epochs)...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = train_correct / total_train

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = val_correct / total_val
        elapsed = time.time() - start_time

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        scheduler.step(epoch_val_loss)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:.2f}%")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            exp1_model_path = settings.MODELS_DIR / "exp1_class_weighted_model.pth"
            torch.save(model.state_dict(), exp1_model_path)
            print(f"  --> Checkpoint saved to '{exp1_model_path}'")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Early Stopping] Triggered at epoch {epoch}")
                break

    # Save Experiment 1 History
    exp1_history_path = settings.MODELS_DIR / "exp1_history.json"
    with open(exp1_history_path, "w") as f:
        json.dump(history, f, indent=2)

    # 4. Evaluate Experiment 1 Model on Untouched Test Set (3,589 samples)
    print("\nEvaluating Experiment 1 Model on Untouched Test Dataset...")
    exp1_model_path = settings.MODELS_DIR / "exp1_class_weighted_model.pth"
    model.load_state_dict(torch.load(exp1_model_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    target_names = [LABEL_MAP[i] for i in range(7)]
    test_acc = float(accuracy_score(all_targets, all_preds))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
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

    exp1_report = {
        "experiment_name": "Experiment 1: Class-Weighted Loss",
        "model": "EmotionCNN",
        "test_accuracy": round(test_acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_score_macro": round(f1_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_score_weighted": round(f1_weighted, 4),
        "train_accuracy_final": round(history["train_acc"][-1], 4),
        "val_accuracy_final": round(history["val_acc"][-1], 4),
        "train_loss_final": round(history["train_loss"][-1], 4),
        "val_loss_final": round(history["val_loss"][-1], 4),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist()
    }

    output_report_path = Path(__file__).resolve().parent / "models_eval" / "exp1_report.json"
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w") as f:
        json.dump(exp1_report, f, indent=2)

    print("\n---------------- EXPERIMENT 1 RESULTS SUMMARY ----------------")
    print(f"Test Accuracy:         {test_acc * 100:.2f}%")
    print(f"Macro F1-Score:        {f1_macro:.4f}")
    print(f"Weighted F1-Score:     {f1_weighted:.4f}")
    print(f"Disgust F1-Score:      {per_class_metrics['Disgust']['f1_score']:.4f}")
    print(f"Happy F1-Score:        {per_class_metrics['Happy']['f1_score']:.4f}")
    print("----------------------------------------------------------------\n")

    return True

if __name__ == "__main__":
    run_experiment_1()
