"""
FER2013 Dataset Loader and Preprocessing Pipelines.
Supports loading FER2013 from fer2013.csv or image directory structures.
Includes Train/Validation/Test splits and Data Augmentation transforms.
"""
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from app.core.config import settings

LABEL_MAP = {
    0: "Surprise",
    1: "Fear",
    2: "Disgust",
    3: "Happy",
    4: "Sad",
    5: "Angry",
    6: "Neutral"
}

# Standard Data Augmentation for training
train_transform = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.RandomCrop(48, padding=4, padding_mode='edge'),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

# Evaluation Transform (No augmentation)
eval_transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

class FER2013CSVDataset(Dataset):
    """Dataset wrapper for FER2013 CSV format (pixels column)."""
    def __init__(self, df: pd.DataFrame, transform=eval_transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        label = int(row['emotion'])
        
        # Parse pixel values
        pixels = np.array(row['pixels'].split(), dtype=np.uint8).reshape(48, 48)
        
        if self.transform:
            image = self.transform(pixels)
        else:
            image = torch.tensor(pixels, dtype=torch.float32).unsqueeze(0) / 255.0

        return image, label

def load_fer2013_data(
    dataset_dir: Path = settings.DATA_DIR / "dataset" / "fer2013",
    batch_size: int = 64,
    val_split: float = 0.15
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Attempts loading FER2013 dataset from directory.
    Returns (train_loader, val_loader, test_loader) if found, else (None, None, None).
    """
    csv_path = dataset_dir / "fer2013.csv"
    
    if csv_path.exists():
        print(f"[Dataset] Found fer2013.csv at {csv_path}. Loading...")
        df = pd.read_csv(csv_path)
        
        # Check usage column if available
        if 'Usage' in df.columns:
            train_df = df[df['Usage'] == 'Training']
            val_df = df[df['Usage'] == 'PublicTest']
            test_df = df[df['Usage'] == 'PrivateTest']
        else:
            # Random split
            num_total = len(df)
            num_val = int(num_total * val_split)
            num_test = int(num_total * val_split)
            num_train = num_total - num_val - num_test
            
            shuffled = df.sample(frac=1.0, random_state=42)
            train_df = shuffled.iloc[:num_train]
            val_df = shuffled.iloc[num_train:num_train+num_val]
            test_df = shuffled.iloc[num_train+num_val:]
            
        train_ds = FER2013CSVDataset(train_df, transform=train_transform)
        val_ds = FER2013CSVDataset(val_df, transform=eval_transform)
        test_ds = FER2013CSVDataset(test_df, transform=eval_transform)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return train_loader, val_loader, test_loader
        
    print(f"[Dataset] No FER2013 dataset found at '{dataset_dir}'.")
    return None, None, None
