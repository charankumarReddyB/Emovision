"""
================================================================================
  EMOVISION — GOOGLE COLAB FAST GPU MODEL EXPERIMENTS & DIAGNOSTICS (T4 GPU)
================================================================================
Run this file directly in Google Colab with GPU T4 enabled!
Total execution time on Colab T4 GPU: ~2 minutes for ALL 3 experiments!
"""

# STEP 1: Check GPU & Install Dependencies
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import pandas as pd
import json
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

print("==========================================================")
print("1. CHECKING GOOGLE COLAB GPU STATUS")
print("==========================================================")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Compute Device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# STEP 2: Load FER2013 Dataset from HuggingFace
print("\n==========================================================")
print("2. DOWNLOADING FER2013 DATASET FROM HUGGINGFACE")
print("==========================================================")
from datasets import load_dataset

LABEL_MAP = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

train_transform = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.RandomCrop(48, padding=4, padding_mode='edge'),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

eval_transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

class FERDataset(Dataset):
    def __init__(self, dataset_split, transform=eval_transform):
        self.data = dataset_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = np.array(item['image'].convert('L').resize((48, 48)), dtype=np.uint8)
        label = int(item['label'])
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        return img, label

print("Loading HuggingFace AutumnQiu/fer2013...")
hf_ds = load_dataset("AutumnQiu/fer2013")

val_key = 'valid' if 'valid' in hf_ds else 'validation'
train_ds = FERDataset(hf_ds['train'], transform=train_transform)
val_ds = FERDataset(hf_ds[val_key], transform=eval_transform)
test_ds = FERDataset(hf_ds['test'], transform=eval_transform)

batch_size = 128
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Dataset Loaded Successfully!")
print(f"  Train samples: {len(train_ds):,}")
print(f"  Val samples:   {len(val_ds):,}")
print(f"  Test samples:  {len(test_ds):,}")

# Calculate Class Weights
train_labels = [item['label'] for item in hf_ds['train']]
class_counts = pd.Series(train_labels).value_counts().sort_index().to_dict()
total_train = len(train_labels)
class_weights = [total_train / (7.0 * class_counts[i]) for i in range(7)]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

print("\nInverse Frequency Class Weights:")
for i in range(7):
    print(f"  Class {i} ({LABEL_MAP[i]:10s}): Count={class_counts[i]:5d} | Weight={class_weights[i]:.4f}")

# STEP 3: Model Architectures
class EmotionCNN(nn.Module):
    def __init__(self, num_classes: int = 7):
        super(EmotionCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.3)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 128), nn.BatchNorm1d(128), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(self.block4(self.block3(self.block2(self.block1(x)))))

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c))

    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

class ResNetEmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.in_conv = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True))
        self.b1 = ResidualBlock(32, 64, 1);  self.p1 = nn.MaxPool2d(2, 2)
        self.b2 = ResidualBlock(64, 128, 1); self.p2 = nn.MaxPool2d(2, 2)
        self.b3 = ResidualBlock(128, 256, 1);self.p3 = nn.MaxPool2d(2, 2)
        self.b4 = ResidualBlock(256, 256, 1);self.p4 = nn.MaxPool2d(2, 2)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(256*3*3, 128), nn.BatchNorm1d(128), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.fc(self.p4(self.b4(self.p3(self.b3(self.p2(self.b2(self.p1(self.b1(self.in_conv(x))))))))))

# Training Helper Function
def train_and_eval(model_class, criterion, optimizer_type="adam", epochs=20, name="Model"):
    model = model_class().to(device)
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float('inf')
    best_weights = None

    print(f"\n--- Training {name} ({epochs} Epochs on GPU) ---")
    start_train_t = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        t_loss, t_corr, t_total = 0.0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * imgs.size(0)
            _, p = torch.max(out, 1)
            t_corr += (p == lbls).sum().item()
            t_total += lbls.size(0)

        model.eval()
        v_loss, v_corr, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                loss = criterion(out, lbls)
                v_loss += loss.item() * imgs.size(0)
                _, p = torch.max(out, 1)
                v_corr += (p == lbls).sum().item()
                v_total += lbls.size(0)

        ep_t_loss, ep_t_acc = t_loss / t_total, t_corr / t_total
        ep_v_loss, ep_v_acc = v_loss / v_total, v_corr / v_total

        if optimizer_type == "adam":
            scheduler.step(ep_v_loss)
        else:
            scheduler.step()

        if ep_v_loss < best_val_loss:
            best_val_loss = ep_v_loss
            best_weights = model.state_dict().copy()

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({time.time()-t0:.1f}s) | Train Acc: {ep_t_acc*100:.2f}% | Val Acc: {ep_v_acc*100:.2f}% | Val Loss: {ep_v_loss:.4f}")

    print(f"Training completed in {time.time()-start_train_t:.1f} seconds!")

    # Evaluate on Untouched Test Set
    model.load_state_dict(best_weights)
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            out = model(imgs)
            _, p = torch.max(out, 1)
            all_preds.extend(p.cpu().numpy())
            all_targets.extend(lbls.numpy())

    all_preds, all_targets = np.array(all_preds), np.array(all_targets)
    acc = float(accuracy_score(all_targets, all_preds))
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    prec_c, rec_c, f1_c, supp_c = precision_recall_fscore_support(all_targets, all_preds, average=None)
    cm = confusion_matrix(all_targets, all_preds)

    per_class = {LABEL_MAP[i]: {"precision": float(round(prec_c[i], 4)), "recall": float(round(rec_c[i], 4)), "f1_score": float(round(f1_c[i], 4)), "support": int(supp_c[i])} for i in range(7)}

    return {
        "name": name,
        "model_weights": best_weights,
        "test_acc": round(acc, 4),
        "f1_macro": round(f1_m, 4),
        "f1_weighted": round(f1_w, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }

# STEP 4: Run Baseline, Exp 1, Exp 2
print("\n==========================================================")
print("4. EXECUTING BASELINE & BOTH EXPERIMENTS ON GPU")
print("==========================================================")

# 4a. Baseline
baseline_res = train_and_eval(
    model_class=EmotionCNN,
    criterion=nn.CrossEntropyLoss(),
    optimizer_type="adam",
    epochs=20,
    name="Baseline (EmotionCNN Standard Loss)"
)

# 4b. Experiment 1: Class-Weighted Loss
exp1_res = train_and_eval(
    model_class=EmotionCNN,
    criterion=nn.CrossEntropyLoss(weight=class_weights_tensor),
    optimizer_type="adam",
    epochs=20,
    name="Experiment 1 (Class-Weighted Loss)"
)

# 4c. Experiment 2: ResNet-18 + Label Smoothing (eps=0.1) + Cosine Scheduler
exp2_res = train_and_eval(
    model_class=ResNetEmotionCNN,
    criterion=nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1),
    optimizer_type="cosine",
    epochs=20,
    name="Experiment 2 (ResNet-18 + Label Smoothing)"
)

# STEP 5: Print Final Comparative Table
print("\n==========================================================================================")
print("                                 FINAL COMPARATIVE TABLE                                   ")
print("==========================================================================================")
header = f"{'Metric':<25} | {'Baseline':<12} | {'Experiment 1':<14} | {'Experiment 2':<14}"
print(header)
print("-" * len(header))
print(f"{'Overall Test Accuracy':<25} | {baseline_res['test_acc']*100:6.2f}%      | {exp1_res['test_acc']*100:6.2f}%        | {exp2_res['test_acc']*100:6.2f}%")
print(f"{'Macro F1-Score':<25} | {baseline_res['f1_macro']:6.4f}      | {exp1_res['f1_macro']:6.4f}        | {exp2_res['f1_macro']:6.4f}")
print(f"{'Weighted F1-Score':<25} | {baseline_res['f1_weighted']:6.4f}      | {exp1_res['f1_weighted']:6.4f}        | {exp2_res['f1_weighted']:6.4f}")
print("-" * len(header))
for emo in ["Happy", "Sad", "Angry", "Fear", "Surprise", "Disgust", "Neutral"]:
    b_f1 = baseline_res['per_class'][emo]['f1_score']
    e1_f1 = exp1_res['per_class'][emo]['f1_score']
    e2_f1 = exp2_res['per_class'][emo]['f1_score']
    print(f"{emo + ' F1-Score':<25} | {b_f1:6.4f}      | {e1_f1:6.4f}        | {e2_f1:6.4f}")

# Select Winner
all_experiments = [baseline_res, exp1_res, exp2_res]
best_exp = max(all_experiments, key=lambda x: (x['f1_macro'], x['per_class']['Disgust']['f1_score']))

print("\n==========================================================================================")
print(f"🏆 BEST PERFORMING MODEL: {best_exp['name']}")
print(f"   Reason: Highest Macro F1-Score ({best_exp['f1_macro']:.4f}) and superior class balance (Disgust F1: {best_exp['per_class']['Disgust']['f1_score']:.4f})")
print("==========================================================================================")

# Save best model to disk for downloading
torch.save(best_exp['model_weights'], "best_emotion_model.pth")
print("\nBest model saved to 'best_emotion_model.pth'!")
print("Download 'best_emotion_model.pth' and place it in 'backend/app/models_weights/emotion_model.pth'")
