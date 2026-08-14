# FER2013 Dataset Directory Setup

Place the **FER2013 Facial Expression Dataset** files in this directory.

### Supported Dataset Formats:

#### Format 1: CSV File (Standard Kaggle FER2013)
Place `fer2013.csv` directly inside this folder:
```text
backend/data/dataset/fer2013/fer2013.csv
```
The CSV file should contain columns:
- `emotion`: Numeric class ID (0 to 6)
- `pixels`: Space-separated pixel values (48x48 = 2304 integers)
- `Usage`: `Training`, `PublicTest`, or `PrivateTest`

#### Format 2: Image Directory Structure
Organize cropped face images into class subfolders:
```text
backend/data/dataset/fer2013/
├── train/
│   ├── happy/
│   ├── sad/
│   ├── angry/
│   ├── fear/
│   ├── surprise/
│   ├── disgust/
│   └── neutral/
└── test/
    ├── happy/
    ├── sad/
    ├── angry/
    ├── fear/
    ├── surprise/
    ├── disgust/
    └── neutral/
```

### Emotion Class Mapping (7 Target Classes):
- `0`: Angry
- `1`: Disgust
- `2`: Fear
- `3`: Happy
- `4`: Sad
- `5`: Surprise
- `6`: Neutral

Once the dataset is placed in this directory, run the training pipeline using:
```powershell
.\backend\.venv\Scripts\python backend/app/ml/train.py
```
