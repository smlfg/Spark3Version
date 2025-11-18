"""Path definitions for the project."""
from pathlib import Path

# Dynamically find project root (assuming this file is in utils/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Define directory paths
MODELS_DIR = PROJECT_ROOT / "models"
DATASETS_DIR = PROJECT_ROOT / "datasets"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Ensure experiments directory exists
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
