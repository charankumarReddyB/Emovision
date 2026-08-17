# Emovision — Real-Time Multi-Face Facial Expression Recognition System

**Capstone Project**  
*High-Accuracy, Real-Time Multi-Face Facial Expression Recognition Using SCRFD-500M ONNX Face Detection, Full Face Crop Alignment, and Official Lightweight EfficientFace (1.27M Params, 88.23% Measured RAF-DB Accuracy) for Fast CPU Live Deployment*

---

## 1. Project Overview

**Emovision** is a high-performance, production-ready computer vision platform designed for real-time facial expression recognition across **N visible faces** simultaneously. 

The system implements a modular deep learning pipeline:
- **High-Quality Face Detection**: Integrates **SCRFD-500M** (`scrfd_500m_bnkps.onnx`) ONNX engine to locate faces and 5 facial keypoints (`left_eye`, `right_eye`, `nose`, `left_mouth`, `right_mouth`).
- **Full Face Crop Alignment**: Warps each detected face into a standardized 224×224 chip with 15% bounding box margin, preserving complete face features (forehead, eyes, nose, mouth, chin).
- **Official EfficientFace Live Model**: Lightweight **EfficientFace** model (`efficientface_repo/checkpoint/efficientface_rafdb.pth`, 1.27M parameters) performing single-pass matrix batch tensor inference (**Measured RAF-DB Test Accuracy: 88.23%**, **196.5 images/sec on CPU**).
- **Official POSTER Reference Model**: High-accuracy **POSTER** model (`poster_repo/checkpoint/rafdb_best.pth`, 71.85M parameters, **92.01% Measured RAF-DB Test Accuracy**) maintained as offline reference/benchmark.
- **FastAPI & WebSocket Telemetry**: Asynchronous WebSocket streaming (`/ws/detection/{session_id}`) broadcasting real-time expression statistics, confidence percentages, and bounding box coordinates to the web frontend.
- **Supabase Cloud Database & Storage**: Production PostgreSQL persistence (`SupabaseRepository`) with local SQLite fallback for session analytics, aggregated stats, and exportable reports.
- **Modern React Single Page Application**: Interactive dashboard featuring live webcam video overlay, facial bounding box overlays, emotion distribution charts, chronological timeline analytics, session logs, and PDF export.

---

## 2. Pipeline Architecture

```text
                                  EMOVISION PIPELINE ARCHITECTURE
                                  
  ┌──────────────┐     ┌────────────────────────────┐     ┌────────────────────────────────┐
  │  Webcam /    │    │ SCRFD-500M ONNX            │    │  5-Point Geometric Affine      │
  │ Video Frame  ├────► Face Detector Engine        ├────► Alignment (224x224 Standard) │
  └──────────────┘     └────────────────────────────┘     └───────────────┬────────────────┘
                                                                           │
  ┌──────────────┐     ┌────────────────────────────┐                     │ N Aligned Chips
  │ React UI     │     │ FastAPI WebSocket Engine   │                     ▼
  │ Live Stream  │◄────┤ Telemetry Broadcast JSON   │◄────────────────────┴────────────────┐
  │ & Dashboard  │     │ & HUD Visual Renderer      │  Official DAN PyTorch Model          │
  └──────────────┘     └────────────────────────────┘  ResNet-18 + Multi-head Attention    │
                                                       Classes: Surprise, Fear, Disgust,   │
                                                       Happy, Sad, Angry, Neutral          │
                                                       └───────────────────────────────────┘
```

---

## 3. Technology Stack

- **Computer Vision & Inference**: Python 3.10+, PyTorch (`torch`), OpenCV, ONNX Runtime (`onnxruntime`), NumPy, SciPy.
- **Backend Infrastructure**: FastAPI, Uvicorn, WebSockets, Asyncio, Pydantic v2, Python-Dotenv.
- **Database & Storage**: Supabase Python SDK, PostgreSQL Cloud Database, SQLite3 local fallback.
- **Frontend Web UI**: React 19, TypeScript, Vite, Tailwind CSS, Lucide React icons, Recharts.
- **Testing & Benchmark Suite**: Pytest, Httpx, End-to-End Multi-Face Benchmark Suite (`cv_benchmark.py`).

---

## 4. Clean Repository Structure

```text
Emovision/
├── .gitignore                    # Excludes node_modules, .venv, caches, and local logs
├── README.md                     # Project documentation & setup instructions
├── backend/                      # Python FastAPI Backend & Computer Vision Core
│   ├── app/
│   │   ├── api/                 # REST & WebSocket endpoints (ws.py, sessions.py, analytics.py)
│   │   ├── core/                # Configuration and global settings (config.py)
│   │   ├── db/                  # Database connections & Supabase repository
│   │   ├── ml/                  # DAN PyTorch network architecture & RAF-DB loader (dan.py)
│   │   ├── models_weights/      # Pretrained models (dan_rafdb.pth, scrfd_500m_bnkps.onnx)
│   │   └── services/            # SCRFD detector, face aligner, & emotion classifier
│   ├── cv_benchmark.py          # Latency & FPS benchmark suite
│   ├── test_model_loading.py    # PyTorch model contract verification test script
│   └── main.py                  # FastAPI application entrypoint
└── frontend/                     # React + Vite + TypeScript Web Frontend
    ├── src/
    │   ├── components/          # Reusable UI components & PDF exporter
    │   ├── screens/             # Live Detection, Analytics, Session History
    │   └── services/            # API & WebSocket client services
    └── package.json             # Frontend dependencies & Vite scripts
```

---

## 5. RAF-DB 7-Class Emotion Mapping

| Index | Emotion Class | Notes |
| :---: | :--- | :--- |
| `0` | **Surprise** | Official DAN RAF-DB Class Index 0 |
| `1` | **Fear** | Official DAN RAF-DB Class Index 1 |
| `2` | **Disgust** | Official DAN RAF-DB Class Index 2 |
| `3` | **Happy** | Official DAN RAF-DB Class Index 3 |
| `4` | **Sad** | Official DAN RAF-DB Class Index 4 |
| `5` | **Angry** | Official DAN RAF-DB Class Index 5 |
| `6` | **Neutral** | Official DAN RAF-DB Class Index 6 |

- **Image Normalization**: ImageNet Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]`
- **Input Dimensions**: `(N, 3, 224, 224)` RGB float tensor
- **Confidence Threshold**: `< 0.50` returns `"Uncertain"`

---

## 6. How to Run the Application

### 1. Run Verification Tests & Benchmarks
```bash
# Navigate to backend directory
cd backend

# Run DAN Model Verification Test
.venv\Scripts\python test_model_loading.py

# Run Latency & FPS Benchmark Suite
.venv\Scripts\python cv_benchmark.py
```

### 2. Start Backend API Server
```bash
cd backend
.venv\Scripts\python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### 3. Start Frontend Web Application
```bash
cd frontend
npm run dev -- --port 5173
```

Open your browser at **`http://localhost:5173`** to access the real-time Live Detection dashboard.
