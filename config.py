
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Flask security
SECRET_KEY = os.environ.get(
    "SECRET_KEY",
    "smartcrop-local-development-key-change-before-deployment"
)

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL")

if DATABASE_URL:
    # Render may provide a URL beginning with postgres://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace(
            "postgres://", "postgresql://", 1
        )

    SQLALCHEMY_DATABASE_URI = DATABASE_URL
else:
    # Local development uses SQLite
    DATABASE_PATH = BASE_DIR / "database" / "smartcrop.db"
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    SQLALCHEMY_DATABASE_URI = (
        f"sqlite:///{DATABASE_PATH.as_posix()}"
    )

SQLALCHEMY_TRACK_MODIFICATIONS = False

# Machine Learning model paths
MODEL_PATH = BASE_DIR / "models" / "crop_model.joblib"
MODEL_INFO_PATH = BASE_DIR / "models" / "model_info.json"
