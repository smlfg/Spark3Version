"""
Path definitions for the Spark3Version project.
"""
from pathlib import Path

# Base directory (Spark3Version root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Asset directories
MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR = BASE_DIR / "datasets"
PROMPTS_DIR = BASE_DIR / "prompts"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
