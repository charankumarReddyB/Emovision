"""
Converts HuggingFace AutumnQiu/fer2013 dataset to Kaggle fer2013.csv format
in backend/data/dataset/fer2013/fer2013.csv.
"""
import os
import time
import numpy as np
import pandas as pd
from datasets import load_dataset

def convert_to_fer2013_csv():
    print("Loading AutumnQiu/fer2013 dataset from HuggingFace...")
    ds = load_dataset("AutumnQiu/fer2013")

    rows = []
    
    # Process Train (Training)
    print(f"Processing Train set ({len(ds['train'])} samples)...")
    for item in ds['train']:
        img_np = np.array(item['image'].convert('L').resize((48, 48)))
        pixel_str = " ".join(map(str, img_np.flatten()))
        rows.append({
            "emotion": int(item['label']),
            "pixels": pixel_str,
            "Usage": "Training"
        })

    # Process Validation (PublicTest)
    val_key = 'valid' if 'valid' in ds else 'validation'
    print(f"Processing Validation set ({len(ds[val_key])} samples)...")
    for item in ds[val_key]:
        img_np = np.array(item['image'].convert('L').resize((48, 48)))
        pixel_str = " ".join(map(str, img_np.flatten()))
        rows.append({
            "emotion": int(item['label']),
            "pixels": pixel_str,
            "Usage": "PublicTest"
        })

    # Process Test (PrivateTest)
    print(f"Processing Test set ({len(ds['test'])} samples)...")
    for item in ds['test']:
        img_np = np.array(item['image'].convert('L').resize((48, 48)))
        pixel_str = " ".join(map(str, img_np.flatten()))
        rows.append({
            "emotion": int(item['label']),
            "pixels": pixel_str,
            "Usage": "PrivateTest"
        })

    df = pd.DataFrame(rows)
    target_dir = os.path.join("data", "dataset", "fer2013")
    os.makedirs(target_dir, exist_ok=True)
    target_csv = os.path.join(target_dir, "fer2013.csv")
    
    print(f"Writing {len(df)} rows to '{target_csv}'...")
    df.to_csv(target_csv, index=False)
    print(f"[SUCCESS] Exported fer2013.csv ({os.path.getsize(target_csv) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    convert_to_fer2013_csv()
