# EmoVision — Real-Time Multi-Face Facial Expression Recognition Platform

**Capstone Project**  
*High-Accuracy, Real-Time Multi-Face Facial Expression Recognition Using SCRFD-500M ONNX Face Detection, Full Face Alignment, and Official EfficientFace (1.27M Params, 88.23% Measured RAF-DB Accuracy) for Fast CPU Live Deployment & Asynchronous Video Analytics.*

---

## 1. Project Overview

**EmoVision** is a high-performance, production-ready computer vision platform designed for real-time facial expression recognition across **multiple visible faces** simultaneously.

The system features:
- **High-Quality Multi-Face Detection**: Integrates **SCRFD-500M** (`scrfd_500m_bnkps.onnx`) ONNX engine with aspect-ratio letterboxing to accurately locate faces and 5 facial keypoints (`left_eye`, `right_eye`, `nose`, `left_mouth`, `right_mouth`).
- **Full Face Crop Alignment**: Warps each detected face into a standardized 224×224 chip with 15% bounding box margin padding.
- **Official EfficientFace Live Model**: Lightweight **EfficientFace** model (`efficientface_rafdb.pth`, 1.27M parameters) performing single-pass matrix batch tensor inference (**Measured RAF-DB Test Accuracy: 88.23%**, **196.5 images/sec on CPU**).
- **Official POSTER Benchmark Model**: High-accuracy **POSTER** model (`rafdb_best.pth`, 71.85M parameters, **92.01% Measured RAF-DB Test Accuracy**) maintained as reference benchmark.
- **3 Analysis Input Modes**:
  1. **Live Webcam**: Asynchronous real-time WebSocket telemetry stream (`/ws/detection/{session_id}`) with live bounding boxes, emotion labels, and confidence overlays.
  2. **Image Upload**: Instant FER analysis for uploaded image files (`.jpg`, `.png`, `.webp`) with Base64 annotated output rendering.
  3. **Video Upload**: Asynchronous background video FER job manager with real progress tracking (`frames_processed / total_frames`), tab-switch non-blocking execution, and browser refresh recovery.
- **Supabase Cloud Database & Storage**: Production PostgreSQL persistence (`SupabaseRepository`) with local SQLite fallback storing session history, source types (`webcam`, `image`, `video`), aggregated statistics, and exportable PDF reports.
- **Modern React Single Page Application**: Interactive UI featuring live webcam stream controls, mode tab selectors, emotion distribution pie charts, timeline analytics, filterable session history, and PDF export.

---

## 2. Pipeline Architecture

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
  │ & Dashboard  │     │ Telemetry Broadcast JSON   │  Official EfficientFace Model        │
  └──────────────┘     └────────────────────────────┘  1.27M Params | 88.23% RAF-DB Acc   │
                                                       Classes: Neutral, Happy, Sad,      │
                                                       Surprise, Fear, Disgust, Angry      │
                                                       └───────────────────────────────────┘
```

---

## 3. Technology Stack

- **Computer Vision & Inference**: Python 3.10+, PyTorch (`torch`), OpenCV, ONNX Runtime (`onnxruntime`), NumPy, SciPy.
- **Backend Infrastructure**: FastAPI, Uvicorn, WebSockets, Asyncio, Pydantic v2, Python-Multipart.
- **Database & Storage**: Supabase Python SDK, PostgreSQL Cloud Database, SQLite3 local fallback.
- **Frontend Web UI**: React 19, TypeScript, Vite, Tailwind CSS, Lucide React icons, Recharts.
- **Testing Suite**: Pytest, TestClient, Httpx, End-to-End Multi-Face Acceptance & Video Lifecycle Test Suites.

---

## 4. Repository Structure

```text
Emovision/
├── README.md                     # Project documentation & setup instructions
├── backend/                      # Python FastAPI Backend & Computer Vision Core
│   ├── app/
│   │   ├── api/                 # REST endpoints (analyze.py, sessions.py, ws.py)
│   │   ├── core/                # Configuration and global settings (config.py)
│   │   ├── db/                  # Database connections & Supabase repository
│   │   ├── models_weights/      # Pretrained models (scrfd_500m_bnkps.onnx)
│   │   └── services/            # SCRFD detector, face aligner, & emotion classifier
│   ├── efficientface_repo/      # EfficientFace model checkpoint & architecture
│   ├── test_acceptance.py       # Multi-face acceptance test suite (1, 2, 3, 5 faces)
│   ├── test_upload_analysis.py  # Image & video upload test suite
│   ├── test_video_lifecycle_and_persistence.py # Async video job & persistence test
│   └── main.py                  # FastAPI application entrypoint
└── frontend/                     # React + Vite + TypeScript Web Frontend
    ├── src/
    │   ├── components/          # Reusable UI components & PDF exporter
    │   ├── screens/             # Live Detection, Image Upload, Video Upload, Analytics, History
    │   └── services/            # API & WebSocket client services
    └── package.json             # Frontend dependencies & Vite scripts
```

---

## 5. RAF-DB 7-Class Emotion Mapping

| Index | Emotion Class | Notes |
| :---: | :--- | :--- |
| `0` | **Neutral** | EfficientFace Index 0 |
| `1` | **Happy** | EfficientFace Index 1 |
| `2` | **Sad** | EfficientFace Index 2 |
| `3` | **Surprise** | EfficientFace Index 3 |
| `4` | **Fear** | EfficientFace Index 4 |
| `5` | **Disgust** | EfficientFace Index 5 |
| `6` | **Angry** | EfficientFace Index 6 |

- **Image Normalization**: Mean `[0.57535914, 0.44928582, 0.40079932]`, Std `[0.20735591, 0.18981615, 0.18132027]`
- **Input Dimensions**: `(N, 3, 224, 224)` RGB float tensor
- **Hardware Execution**: Optimized 8-thread CPU batch processing

---

## 6. How to Run Locally

### 1. Run Verification Test Suites
```bash
# Navigate to backend directory
cd backend

# Run Multi-Face Detection Acceptance Suite (1, 2, 3, 5 faces)
.venv\Scripts\python test_acceptance.py

# Run Asynchronous Video Lifecycle & Supabase Persistence Test
.venv\Scripts\python test_video_lifecycle_and_persistence.py
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

Open your browser at **`http://localhost:5173`**.

---

## 7. Deployment Instructions

### Frontend (Netlify / Vercel)
1. Set Build Command: `npm run build`
2. Set Publish Directory: `dist`
3. Environment Variables:
   - `VITE_API_URL`: Your backend API server URL (e.g. `https://your-emovision-backend.onrender.com`)

### Backend (Render / Railway / Cloud Run)
1. Python Version: `3.10+`
2. Start Command: `uvicorn main:app --host 0.0.0.0 --port 8000`
3. Install system libraries: `libgl1-mesa-glx` (for OpenCV)
