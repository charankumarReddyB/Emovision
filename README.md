# Emovision — Real-Time Multi-Face Facial Expression Recognition System

**Capstone Project**  
*High-Accuracy, Real-Time Multi-Face Facial Expression Recognition Using SCRFD / YuNet ONNX Face Detection, 5-Point Affine Face Alignment, and MobileFaceNet Deep Learning*

---

## 1. Project Overview

**Emovision** is a high-performance, production-ready computer vision platform designed for real-time facial expression recognition across **N visible faces** simultaneously. 

The system implements a modular deep learning pipeline:
- **High-Quality Face Detection**: Integrates **SCRFD-500M** (`scrfd_500m_bnkps.onnx`) and OpenCV **YuNet** (`face_detection_yunet_2023mar.onnx`) ONNX engines to locate faces and 5 facial keypoints (`left_eye`, `right_eye`, `nose`, `left_mouth`, `right_mouth`).
- **Geometric 5-Point Affine Alignment**: Warps each detected face into a standardized 112×112 BGR chip using landmark affine transformations, preserving eye angle horizontal alignment for accurate feature extraction.
- **MobileFaceNet ONNX Emotion Classifier**: OpenCV Zoo MobileFaceNet FER model (`facial_expression_recognition_mobilefacenet_2022july.onnx`) performing batch ONNX inference across 7 target expression classes with softmax probability normalization.
- **FastAPI & WebSocket Telemetry**: Asynchronous WebSocket streaming (`/ws/detection/{session_id}`) broadcasting real-time expression statistics, confidence percentages, and bounding box coordinates to the web frontend.
- **Supabase Cloud Database & Storage**: Production PostgreSQL persistence (`SupabaseRepository`) with local SQLite fallback for session analytics, aggregated stats, and exportable reports.
- **Modern React Single Page Application**: Interactive dashboard featuring live webcam video overlay, facial bounding box overlays, emotion distribution charts, chronological timeline analytics, and session logs.

---

## 2. Pipeline Architecture

```
                                  EMOVISION PIPELINE ARCHITECTURE
                                  
  ┌──────────────┐     ┌────────────────────────────┐     ┌────────────────────────────────┐
  │  Webcam /    │    │ SCRFD-500M / OpenCV YuNet │    │  5-Point Geometric Affine      │
  │ Video Frame  ├────► ONNX Face Detector Engine   ├────► Alignment (112x112 Standard) │
  └──────────────┘     └────────────────────────────┘     └───────────────┬────────────────┘
                                                                          │
  ┌──────────────┐     ┌────────────────────────────┐                     │ N Aligned Chips
  │ React UI     │     │ FastAPI WebSocket Engine   │                     ▼
  │ Live Stream  │◄────┤ Telemetry Broadcast JSON   │◄────────────────────┴────────────────┐
  │ & Dashboard  │     │ & HUD Visual Renderer      │  MobileFaceNet ONNX FER Model        │
  └──────────────┘     └────────────────────────────┘  7 Classes: Happy, Sad, Angry, Fear, │
                                                       Surprise, Disgust, Neutral              │
                                                       └───────────────────────────────────────┘
```

---

## 3. Technology Stack

- **Computer Vision & Inference**: Python 3.10+, OpenCV, ONNX Runtime (`onnxruntime`), NumPy, SciPy.
- **Backend Infrastructure**: FastAPI, Uvicorn, WebSockets, Asyncio, Pydantic v2, Python-Dotenv.
- **Database & Storage**: Supabase Python SDK, PostgreSQL Cloud Database, SQLite3 local fallback.
- **Frontend Web UI**: React 19, TypeScript, Vite, Tailwind CSS, Lucide React icons, Recharts.
- **Testing & Quality Assurance**: Pytest, Httpx, End-to-End Multi-Face Benchmark Suite.

---

## 4. Repository Structure

```
Emovision/
├── backend/
│   ├── app/
│   │   ├── api/                 # FastAPI REST & WebSocket routers (health, sessions, analytics, ws)
│   │   ├── core/                # Configuration and global settings (config.py)
│   │   ├── db/                  # Database abstraction layer & Supabase integration
│   │   ├── models_weights/      # Pretrained ONNX model weights (SCRFD, YuNet, MobileFaceNet)
│   │   ├── schemas/             # Pydantic data validation schemas
│   │   └── services/            # Face detection, 5-point alignment, ONNX classifier, pipeline services
│   ├── data/                    # Local SQLite fallback database storage
│   ├── tests/                   # Pytest automated test suite
│   ├── main.py                  # FastAPI server entrypoint
│   ├── test_full_suite.py       # E2E benchmark & test suite
│   └── requirements.txt         # Backend Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/          # Reusable UI layout & navigation components
│   │   ├── screens/             # Dashboard, LiveDetection, Analytics, SessionHistory
│   │   ├── services/            # REST API client & WebSocket streaming service
│   │   ├── types.ts             # TypeScript interface definitions
│   │   └── App.tsx              # Main routing & application entrypoint
│   ├── package.json             # React dependencies and scripts
│   └── vite.config.ts           # Vite build configuration
├── .gitignore                   # Version control exclusion rules
└── README.md                    # Project documentation
```

---

## 5. Prerequisites & Environment Setup

### System Requirements
- **Python**: 3.10 or higher
- **Node.js**: v18.0 or higher
- **Git**: Installed and configured

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/charankumarReddyB/Emovision.git
   cd Emovision
   ```

2. **Backend Setup**:
   ```bash
   cd backend
   python -m venv .venv
   
   # Windows PowerShell / CMD:
   .venv\Scripts\activate
   
   # Linux / macOS:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Frontend Setup**:
   ```bash
   cd ../frontend
   npm install
   ```

---

## 6. Environment Configuration

Create a `.env` file in the `backend/` directory:

```env
# Database configuration ('supabase' or 'sqlite')
DATABASE_TYPE=supabase

# Supabase Credentials
SUPABASE_URL=https://amvsouizmnzqkvpvegcu.supabase.co
SUPABASE_KEY=your-supabase-anon-or-service-role-key

# Server settings
DEBUG=False
VERSION=1.0.0
PORT=8000
```

---

## 7. Running the Application

### 1. Start FastAPI Backend Server
```bash
cd backend
.venv\Scripts\python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
- API Base URL: `http://127.0.0.1:8000`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`

### 2. Start React Frontend Client
```bash
cd frontend
npm run dev
```
- Web Application: `http://localhost:5173`

---

## 8. API & WebSocket Endpoint Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `GET /api/health` | `GET` | Health check returning status, database connectivity, and UTC timestamp. |
| `GET /api/model` | `GET` | Model metadata, active ONNX detector, and target 7 facial emotion classes. |
| `POST /api/session/start` | `POST` | Initializes a new live tracking session and returns `session_id`. |
| `GET /api/session/{id}/current` | `GET` | Returns latest frame face bounding boxes, keypoints, emotions, and confidence. |
| `POST /api/session/{id}/end` | `POST` | Finalizes tracking session, calculates aggregate metrics, and saves stats to DB. |
| `GET /api/session/{id}/analytics` | `GET` | Aggregated analytics, dominant emotion counts, and facial telemetry. |
| `GET /api/sessions` | `GET` | Paginated list of historical detection sessions. |
| `WS /ws/detection/{session_id}` | `WS` | Real-time WebSocket connection streaming live detection frames & statistics. |

---

## 9. Model Details & Performance

- **Face Detection Engine**: SCRFD-500M (`scrfd_500m_bnkps.onnx`) & OpenCV YuNet (`face_detection_yunet_2023mar.onnx`) with 5 facial keypoints (`left_eye`, `right_eye`, `nose`, `left_mouth`, `right_mouth`).
- **Face Alignment**: 5-point similarity transformation warping raw face bounding boxes to standardized 112×112 BGR chips.
- **Emotion Classifier**: OpenCV Zoo MobileFaceNet FER ONNX Model (`facial_expression_recognition_mobilefacenet_2022july.onnx`) evaluating 7 classes (`Angry`, `Disgust`, `Fear`, `Happy`, `Neutral`, `Sad`, `Surprise`).
- **Inference Latency**: `~38 ms per face (~26 FPS)` on standard CPU hardware.

---

## 10. Automated Testing

Run backend tests using Pytest and the integrated benchmark suite:

```bash
# Run unit & API integration tests
cd backend
.venv\Scripts\pytest tests/ -v

# Run full end-to-end multi-face benchmark suite
.venv\Scripts\python test_full_suite.py
```

---

## 11. License & Capstone Credits

Developed as part of the Computer Vision Capstone Project on **Real-Time Human Emotion Recognition Using Facial Expressions**.
