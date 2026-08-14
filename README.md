# Emovision — Computer Vision-Based Real-Time Human Emotion Recognition Platform

**B.Tech Capstone Project**  
*Computer Vision-Based Real-Time Human Emotion Recognition Using Facial Expressions*

---

## 1. Project Overview

**Emovision** is an end-to-end, multi-person facial expression recognition platform capable of dynamically detecting, tracking, and classifying facial emotions across **N visible people** in real time. Built with OpenCV, PyTorch/Keras lightweight CNNs, FastAPI, WebSocket streaming, Supabase PostgreSQL, and React + Tailwind CSS, the platform delivers 100+ FPS real-time emotion telemetry, temporal prediction smoothing, individual Person ID metrics, and rich interactive analytics dashboards.

---

## 2. Key Features

- **Dynamic N-Person Tracking**: OpenCV face detector paired with a spatial Centroid & IoU tracker that assigns persistent, unique Person IDs (`Person 1`, `Person 2`, etc.) to visible faces.
- **7 Target Facial Expression Classes**: Classifies faces into `Happy`, `Sad`, `Angry`, `Fear`, `Surprise`, `Disgust`, and `Neutral`.
- **Temporal Prediction Smoothing**: Moving-window majority voting queue suppresses single-frame prediction flicker and noise.
- **Real-Time WebSocket Telemetry**: Streams structured detection JSON frames (`person_id`, `expression`, `confidence`, `bounding_box`) at 12–16 Hz.
- **Production Supabase PostgreSQL Database**: Cloud-hosted PostgreSQL persistence with asynchronous background logging for zero CV pipeline latency overhead, plus local SQLite fallback.
- **Interactive React Figma Frontend**: Four screens — Dashboard (KPI cards, distribution donut chart, frequency bar chart), Live Detection (real-time video bounding box overlays & gauges), Analytics (person-wise selector, chronological emotion timelines, timeline area chart), and Session History (paginated inspection logs & detail modals).

---

## 3. System Architecture

```
React Frontend (Vite + TS + Tailwind CSS)
       │
       ▼  (REST APIs / WebSocket Stream)
FastAPI Application Layer (Python 3.14)
       │
       ├────► OpenCV Face Detection & Centroid Tracker (Person IDs)
       ├────► Lightweight EmotionCNN Inference Engine (PyTorch)
       ├────► Temporal Prediction Queue Smoother
       │
       ▼  (Asynchronous Thread Workers)
Database Access Layer (DatabaseRepository Abstraction)
       │
       ├────► Supabase PostgreSQL (Production Cloud Database)
       └────► SQLite Database (`data/emovision.db` Local Fallback)
```

---

## 4. Technology Stack

- **Computer Vision & ML**: Python 3.14, OpenCV, PyTorch / Keras, NumPy, Pandas.
- **Backend API**: FastAPI, Uvicorn, WebSockets, Pydantic v2, Python-Dotenv.
- **Database Layer**: Supabase Python SDK, PostgreSQL, SQLAlchemy ORM, SQLite3.
- **Frontend UI**: React 19, TypeScript, Vite 8, Tailwind CSS v4, Lucide React, Recharts.
- **Testing & Quality**: Pytest, Httpx, Custom E2E Scaling Benchmark Suite.

---

## 5. Project Structure

```
Emovision/
├── backend/
│   ├── app/
│   │   ├── api/                 # FastAPI REST Endpoint Routers (health, sessions, analytics)
│   │   ├── core/                # Global Settings & Config (config.py)
│   │   ├── db/                  # Database Layer (repository.py, database.py, orm_models.py)
│   │   ├── models_weights/      # Pretrained CNN Weights (emotion_model.pth)
│   │   ├── schemas/             # Pydantic Request/Response Schemas (api_schemas.py)
│   │   └── services/            # CV & Pipeline Services (face_detector, face_tracker, emotion_classifier)
│   ├── data/                    # Local SQLite Database Storage (emovision.db)
│   ├── tests/                   # Pytest Unit Test Suite (28 unit tests)
│   ├── .env.example             # Environment configuration template
│   ├── main.py                  # FastAPI Application Entrypoint
│   ├── migrate_sqlite_to_supabase.py # Data Migration Script
│   ├── supabase_schema.sql      # PostgreSQL DDL Schema Script
│   ├── test_full_suite.py       # E2E Benchmark Test Suite
│   └── verify_supabase_live_data.py # Supabase Live Verification Tool
├── frontend/
│   ├── src/
│   │   ├── components/          # Layout & UI Components (Navbar, Header)
│   │   ├── screens/             # Dashboard, LiveDetection, Analytics, SessionHistory
│   │   ├── services/            # REST API Client (api.ts) & WebSocket Manager (websocket.ts)
│   │   ├── types.ts             # TypeScript Application Interfaces
│   │   └── App.tsx              # Main Navigation & Screen Routing
│   ├── package.json             # React Dependencies & Scripts
│   └── vite.config.ts           # Vite Build Configuration
├── .gitignore                   # Security & Artifact Exclusion Policy
└── README.md                    # Project Documentation
```

---

## 6. Installation & Prerequisites

### Prerequisites
- Python 3.10+ (Python 3.14 recommended)
- Node.js 18+ and npm
- Git

### Clone Repository
```bash
git clone https://github.com/charankumarReddyB/Emovision.git
cd Emovision
```

---

## 7. Environment Variables Configuration

Create a `.env` file in the `backend/` directory:

```env
# Database Type: 'supabase' for production PostgreSQL, 'sqlite' for local fallback
DATABASE_TYPE=supabase

# Supabase Production Credentials
SUPABASE_URL=https://amvsouizmnzqkvpvegcu.supabase.co
SUPABASE_KEY=your-supabase-service-role-or-anon-key

# Application Settings
DEBUG=False
VERSION=1.0.0
PORT=8000
```

*Template reference available in `backend/.env.example`.*

---

## 8. Supabase Setup Instructions

1. Log in to [Supabase](https://supabase.com) and open your project dashboard.
2. Go to **Project Settings → API** to copy your `Project URL` and `service_role` secret key into `backend/.env`.
3. Open **SQL Editor**, click **New Query**, and execute the DDL script from `backend/supabase_schema.sql`:
   ```sql
   -- Creates sessions and predictions tables, foreign keys, indexes, and RLS policies
   ```
4. *(Optional)* Migrate local SQLite history to Supabase:
   ```bash
   cd backend
   .venv\Scripts\python migrate_sqlite_to_supabase.py
   ```

---

## 9. Backend Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 10. Frontend Setup

```bash
cd frontend
npm install
```

---

## 11. How to Run the Application

### 1. Launch FastAPI Backend
```bash
cd backend
.venv\Scripts\python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
- REST API Server: `http://127.0.0.1:8000`
- Interactive OpenAPI Swagger Docs: `http://127.0.0.1:8000/docs`

### 2. Launch React Frontend
```bash
cd frontend
cmd /c npm run dev
```
- Web Client Application: `http://localhost:5173`

---

## 12. API Documentation Summary

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `GET /api/health` | `GET` | System, CV model, tracking status, and server UTC timestamp. |
| `GET /api/model/info` | `GET` | Emotion model name, 7 target classes, and input face dimensions `[48, 48]`. |
| `POST /api/session/start` | `POST` | Initializes a new tracking session and returns `session_id`. |
| `GET /api/session/{id}/current` | `GET` | Returns latest frame face bounding boxes, emotions, and confidence scores. |
| `POST /api/session/{id}/end` | `POST` | Finalizes session, computes duration, prediction totals, and updates DB. |
| `GET /api/session/{id}/analytics` | `GET` | Returns aggregate metrics, emotion distribution, and Person ID list. |
| `GET /api/session/{id}/person/{p_id}` | `GET` | Returns individual metrics and chronological emotion timeline for Person ID. |
| `GET /api/sessions` | `GET` | Returns paginated historical session logs (`page`, `limit`). |
| `GET /api/sessions/{session_id}` | `GET` | Inspection details for a selected session. |
| `WS /ws/detection/{session_id}` | `WS` | Real-time WebSocket stream delivering 12–16 Hz detection JSON frames. |

---

## 13. Model Architecture & Pipeline Information

- **Input Dimension**: `(48, 48, 1)` Grayscale face chips normalized to `[0.0, 1.0]`.
- **Classifier**: Lightweight 4-block Convolutional Neural Network (`EmotionCNN`) with BatchNorm, MaxPool2d, Dropout (0.25/0.5), and 7 Dense linear outputs.
- **Face Detection**: OpenCV Haar Cascade / DNN face detector extracting bounding boxes `(x, y, w, h)`.
- **Multi-Person Tracking**: Centroid spatial distance & IoU overlap tracker maintaining persistent Person IDs across frame gaps up to 30 frames.

---

## 14. Empirical Performance & Benchmarks

All metrics were measured empirically on host hardware:

- **Model Forward Pass Latency**: **2.20 ms** per face chip
- **1-Face Pipeline Latency**: **9.80 ms** (**102.1 FPS**)
- **2-Face Pipeline Latency**: **8.93 ms** (**111.9 FPS**)
- **3-Face Pipeline Latency**: **7.85 ms** (**127.3 FPS**)
- **5-Face Pipeline Latency**: **9.90 ms** (**101.0 FPS**)
- **8-Face Pipeline Latency**: **13.40 ms** (**74.6 FPS**)
- **WebSocket Broadcast Rate**: **12.4 – 15.6 Hz**
- **REST API Latency**: **2.9 ms – 18.5 ms**

---

## 15. Testing & Verification Information

Run automated unit and integration tests:

```bash
# 1. Run Pytest Unit Test Suite (28 unit tests)
.\backend\.venv\Scripts\pytest backend/tests/ -v

# 2. Run Comprehensive Live E2E Benchmark Suite
cd backend && .venv\Scripts\python test_full_suite.py

# 3. Run Live Supabase Data Audit Script
cd backend && .venv\Scripts\python verify_supabase_live_data.py
```
