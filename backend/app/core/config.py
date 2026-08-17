"""
Emovision Backend Configuration Module.
Contains global settings, model configurations, and default paths.
"""
import os
from pathlib import Path
from pydantic import BaseModel, Field

try:
    import dotenv
    dotenv.load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except Exception:
    pass

# Base Directory Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR_PATH = BASE_DIR / "app" / "models_weights"
DATA_DIR_PATH = BASE_DIR / "data"

# Ensure runtime directories exist
DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
MODELS_DIR_PATH.mkdir(parents=True, exist_ok=True)

class Settings(BaseModel):
    PROJECT_NAME: str = "Emovision Backend"
    VERSION: str = "1.0.0"
    # Storage & Database Settings
    DATABASE_TYPE: str = Field(default_factory=lambda: os.getenv("DATABASE_TYPE", "supabase").lower())
    SUPABASE_URL: str = Field(default_factory=lambda: os.getenv("SUPABASE_URL", ""))
    SUPABASE_KEY: str = Field(default_factory=lambda: os.getenv("SUPABASE_KEY", ""))
    DATABASE_URL: str = Field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    DATABASE_PATH: Path = DATA_DIR_PATH / "emovision.db"
    # Root Models Directory & Colab Integration Contract
    ROOT_MODELS_DIR: Path = BASE_DIR.parent / "models"
    EMOTION_MODEL_NAME: str = "emotion_model.onnx"
    FALLBACK_MODEL_NAME: str = "facial_expression_recognition_mobilefacenet_2022july.onnx"
    
    # Emotion Confidence Threshold (display 'Uncertain' if confidence < 0.50)
    CONFIDENCE_THRESHOLD: float = Field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.50")))

settings = Settings()
