"""
Training Pipeline for Facial Expression Recognition (FER) Model.
Trains EmotionCNN on FER2013 dataset using PyTorch with Data Augmentation,
Validation, Early Stopping, and Checkpointing.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import time
import sys

# Add backend directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.core.config import settings
from app.ml.model import EmotionCNN
from app.ml.dataset import load_fer2013_data, LABEL_MAP

def train_model(
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    patience: int = 7
):
    """
    Executes model training pipeline if dataset exists.
    """
    print("\n=======================================================")
    print("EMOVISION FACIAL EXPRESSION RECOGNITION - TRAINING PIPELINE")
    print("=======================================================")
    
    # 1. Load Dataset
    train_loader, val_loader, test_loader = load_fer2013_data(batch_size=batch_size)
    
    if train_loader is None:
        dataset_dir = settings.DATA_DIR / "dataset" / "fer2013"
        print(f"\n[DATASET MISSING ERROR]")
        print(f"The FER2013 dataset was NOT found at: '{dataset_dir}'")
        print(f"To train the model:")
        print(f"  1. Place 'fer2013.csv' in '{dataset_dir / 'fer2013.csv'}'")
        print(f"  2. Refer to '{dataset_dir / 'README.md'}' for detailed instructions.")
        print(f"  3. Re-run this script: python backend/app/ml/train.py")
        print(f"\nTraining pipeline halted safely without inventing fake results.")
        return False

    import os
    if not torch.cuda.is_available():
        torch.set_num_threads(os.cpu_count() or 8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} (CPU Threads: {torch.get_num_threads()})")

    # 2. Instantiate Model, Loss, Optimizer & Scheduler
    model = EmotionCNN(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\nStarting Training ({epochs} Max Epochs, Batch Size: {batch_size}, Init LR: {learning_rate})...\n")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # --- Training Phase ---
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

        # --- Validation Phase ---
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

        # Update History
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        scheduler.step(epoch_val_loss)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:.2f}%")

        # --- Checkpointing & Early Stopping ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            
            # Save Model Checkpoint
            model_save_path = settings.MODELS_DIR / "emotion_model.pth"
            torch.save(model.state_dict(), model_save_path)
            
            # Save Label Mapping & History
            labels_save_path = settings.MODELS_DIR / "emotion_labels.json"
            with open(labels_save_path, "w") as f:
                json.dump({
                    "labels": LABEL_MAP,
                    "target_emotions": settings.EMOTION_CLASSES,
                    "best_val_acc": round(epoch_val_acc, 4)
                }, f, indent=2)
                
            print(f"  --> Checkpoint saved to '{model_save_path}' (Val Acc: {epoch_val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] No improvement in validation loss for {patience} consecutive epochs.")
                break

    # Save training history
    history_save_path = settings.MODELS_DIR / "training_history.json"
    with open(history_save_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  --> Training history saved to '{history_save_path}'")

    print("\nTraining Completed Successfully!")
    return True

if __name__ == "__main__":
    train_model()
