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
    # Models Directory & Model Integration Contract
    ROOT_MODELS_DIR: Path = MODELS_DIR_PATH
    MODELS_DIR: Path = MODELS_DIR_PATH
    EMOTION_MODEL_NAME: str = "dan_rafdb.pth"
    ONNX_MODEL_NAME: str = "emotion_model.onnx"
    
    # Face Preprocessing Settings
    TARGET_FACE_SIZE: tuple[int, int] = (224, 224)
    COLOR_MODE: str = "rgb"
    
    # Official DAN RAF-DB Emotion Classes (Label Order: 0->Surprise, 1->Fear, 2->Disgust, 3->Happy, 4->Sad, 5->Angry, 6->Neutral)
    EMOTION_CLASSES: list[str] = [
        "Surprise",
        "Fear",
        "Disgust",
        "Happy",
        "Sad",
        "Angry",
        "Neutral"
    ]
    
    # Emotion Confidence Threshold (display 'Uncertain' if confidence < 0.50)
    CONFIDENCE_THRESHOLD: float = Field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.50")))

settings = Settings()
