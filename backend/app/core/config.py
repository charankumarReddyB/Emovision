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
    MODELS_DIR: Path = MODELS_DIR_PATH
    DATA_DIR: Path = DATA_DIR_PATH
    
    # Face Detection Settings
    DETECTOR_TYPE: str = "yunet_haar"  # 'yunet_haar', 'haar', 'dnn'
    DETECTION_MIN_CONFIDENCE: float = 0.5
    INPUT_WIDTH: int = 640
    INPUT_HEIGHT: int = 480
    
    # Face Preprocessing Settings
    TARGET_FACE_SIZE: tuple[int, int] = (48, 48)  # Standard input for emotion recognition CNNs
    COLOR_MODE: str = "grayscale"  # 'grayscale' or 'rgb'
    
    # Multi-Face Tracking Settings
    MAX_DISAPPEARED_FRAMES: int = 30  # Max frames an ID remains active without detection
    IOU_THRESHOLD: float = 0.3         # Min IoU for matching faces across frames
    MAX_CENTROID_DISTANCE: float = 100.0  # Max spatial pixel distance for tracking match
    
    # Target Emotion Classes (FER Standard)
    EMOTION_CLASSES: list[str] = [
        "Happy",
        "Sad",
        "Angry",
        "Fear",
        "Surprise",
        "Disgust",
        "Neutral"
    ]

settings = Settings()
