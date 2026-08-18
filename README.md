# EmoVision — Real-Time Multi-Face Facial Expression Recognition Platform

**Capstone Project**  
*High-Accuracy, Real-Time Multi-Face Facial Expression Recognition Using SCRFD-500M ONNX Face Detection, 5-Point Geometric Affine Face Alignment, and Multi-Model Architectures (EfficientFace, DAN, EmotionCNN) Supported across RAF-DB and FER2013 Datasets.*

---

## 1. Project Overview

**EmoVision** is a high-performance, production-ready computer vision platform designed for real-time facial expression recognition across **multiple visible faces** simultaneously.

The platform supports multiple model architectures and dataset pipelines:

### Primary Models & Detectors
- **Primary Face Detector**: **SCRFD-500M** (`scrfd_500m_bnkps.onnx`) ONNX engine with letterbox padding, high NMS IoU filtering, and 5 facial keypoint detection (`left_eye`, `right_eye`, `nose`, `left_mouth`, `right_mouth`).
- **Primary Live Emotion Model**: Lightweight **EfficientFace** (`efficientface_rafdb.pth`, 1.27M parameters, **88.23% RAF-DB accuracy**, **196.5 frames/sec on CPU**).

### Alternative Models & Training Pipelines Included
- **DAN (Distract Your Attention Network)**: Multi-head cross-attention FER model (`app/ml/dan.py`, **89.70% benchmark accuracy**).
- **EmotionCNN**: Custom 4-layer PyTorch convolutional neural network (`app/ml/model.py`) with BatchNorm, Dropout (0.4), and MaxPool.
- **FER2013 & RAF-DB Datasets**: Dataset processing utilities (`app/ml/dataset.py`, `prepare_fer2013_csv.py`) supporting both Kaggle FER2013 (48x48 grayscale, 35,887 images) and RAF-DB (224x224 RGB, 15,339 images).
- **Training & Transfer Learning Experiments**: Included scripts for class-weighted cross-entropy loss, focal loss, and transfer learning comparisons (`run_exp1_class_weighted.py`, `run_exp2_improved_model.py`, `run_transfer_learning_experiments.py`, `eval_and_generate_plots.py`).

---

## 2. Supported Datasets & 7-Class Emotion Mappings

### RAF-DB 7 Basic Emotion Classes
| Index | Emotion Class | RAF-DB Label | EfficientFace Mapping |
| :---: | :--- | :--- | :--- |
| `0` | **Neutral** | Neutral | Index 0 |
| `1` | **Happy** | Happy | Index 1 |
| `2` | **Sad** | Sad | Index 2 |
| `3` | **Surprise** | Surprise | Index 3 |
| `4` | **Fear** | Fear | Index 4 |
| `5` | **Disgust** | Disgust | Index 5 |
| `6` | **Angry** | Angry | Index 6 |

### FER2013 7 Emotion Classes
| Index | Class Name | FER2013 Standard Index |
| :---: | :--- | :--- |
| `0` | **Angry** | Index 0 |
| `1` | **Disgust** | Index 1 |
| `2` | **Fear** | Index 2 |
| `3` | **Happy** | Index 3 |
| `4` | **Sad** | Index 4 |
| `5` | **Surprise** | Index 5 |
| `6` | **Neutral** | Index 6 |

---

## 3. Pipeline Architecture

```text
                                  EMOVISION PIPELINE ARCHITECTURE
                                  
  ┌──────────────┐     ┌────────────────────────────┐     ┌────────────────────────────────┐
  │  Webcam /    │    │ SCRFD-500M ONNX            │    │  5-Point Geometric Affine      │
  │ Image / Video├────► Face Detector Engine        ├────► Alignment (224x224 Standard) │
  └──────────────┘     └────────────────────────────┘     └───────────────┬────────────────┘
                                                                             │
  ┌──────────────┐     ┌────────────────────────────┐                     │ N Aligned Chips
  │ React UI     │     │ FastAPI REST / WS Engine   │                     ▼
  │ Live Stream  │◄────┤ Asynchronous Job Manager   │◄────────────────────┴────────────────┐
  │ & Dashboard  │     │ Telemetry Broadcast JSON   │  Multi-Model Emotion Classification  │
  └──────────────┘     └────────────────────────────┘  • EfficientFace (1.27M, 88.23% RAF) │
                                                       • DAN Cross-Attention (89.70% RAF) │
                                                       • EmotionCNN (FER2013 7-Class)    │
                                                       └───────────────────────────────────┘
```

---

## 4. Repository Structure

```text
Emovision/
├── README.md                     # Comprehensive project documentation
├── backend/                      # Python FastAPI Backend & Computer Vision Engine
│   ├── app/
│   │   ├── api/                 # REST endpoints (ws.py, analyze.py, session.py, analytics.py)
│   │   ├── core/                # Configuration and global settings (config.py)
│   │   ├── db/                  # Database repository (Supabase PostgreSQL + SQLite fallback)
│   │   ├── ml/                  # Alternative model classes & dataset utilities
│   │   │   ├── dan.py           # DAN (Distract Your Attention) model architecture
│   │   │   ├── model.py         # Custom EmotionCNN PyTorch model
│   │   │   ├── dataset.py       # FER2013 dataset loader and preprocessing
│   │   │   ├── train.py         # PyTorch model training loop
│   │   │   └── evaluate.py      # Confusion matrix & metrics generator
│   │   ├── models_weights/      # Pretrained ONNX weights (scrfd_500m_bnkps.onnx)
│   │   └── services/            # RealtimePipeline, EfficientFace, SCRFD, FaceAligner
│   ├── efficientface_repo/      # Official EfficientFace model checkpoint & architecture
│   ├── prepare_fer2013_csv.py   # FER2013 dataset preparation script
│   ├── run_exp1_class_weighted.py # Experiment 1 (Class-Weighted Loss)
│   ├── run_exp2_improved_model.py # Experiment 2 (Focal Loss & Data Augmentation)
│   ├── run_transfer_learning_experiments.py # Transfer learning suite
│   ├── eval_and_generate_plots.py # Plot and metrics generator
│   ├── test_full_suite.py       # Master automated test suite
│   ├── test_acceptance.py       # Multi-face detection acceptance tests
│   ├── test_upload_analysis.py  # Image & video upload test suite
│   ├── test_video_lifecycle_and_persistence.py # Async video job test
│   └── main.py                  # FastAPI application entrypoint
└── frontend/                     # React + Vite + TypeScript Web Frontend
    ├── src/
    │   ├── components/          # Reusable UI components & PDF report generator
    │   ├── screens/             # Live Detection, Image Upload, Video Upload, Analytics, History
    │   ├── services/            # API & WebSocket client services
    │   └── utils/               # Timezone-aware date formatting (Asia/Kolkata)
    └── package.json             # Frontend dependencies & Vite scripts
```

---

## 5. Technology Stack

- **Computer Vision & Deep Learning**: PyTorch (`torch`), torchvision, OpenCV, ONNX Runtime (`onnxruntime`), NumPy, SciPy, timm.
- **Backend Infrastructure**: FastAPI, Uvicorn, WebSockets, Asyncio, Pydantic v2, Python-Multipart, SQLAlchemy.
- **Database & Storage**: Supabase Python SDK, PostgreSQL Cloud Database, SQLite3 local fallback.
- **Frontend Web UI**: React 19, TypeScript, Vite, Tailwind CSS, Lucide React icons, Recharts.
- **Testing & Benchmarks**: Pytest, TestClient, Httpx, Acceptance & Integration Test Suites.

---

## 6. How to Run Locally

### 1. Start Backend API Server
```bash
cd backend
py -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### 2. Start Frontend Web Application
```bash
cd frontend
npm run dev -- --port 5173
```

Open your browser at **`http://localhost:5173`**.

### 3. Run Automated Regression Test Suite
```bash
cd backend
py test_full_suite.py
```
