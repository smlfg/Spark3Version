from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
DATASETS_DIR = PROJECT_ROOT / "datasets"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
