"""Path utilities for Spark3Version project."""

from pathlib import Path


# Project root directory (parent of Spark3Version package)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Main package directory
PACKAGE_ROOT = PROJECT_ROOT / "Spark3Version"

# Core directories
MODELS_DIR = PACKAGE_ROOT / "models"
DATASETS_DIR = PACKAGE_ROOT / "datasets"
PROMPTS_DIR = PACKAGE_ROOT / "prompts"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path to ensure exists

    Returns:
        The path object
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_model_dir(model_name: str) -> Path:
    """Get the directory path for a specific model.

    Args:
        model_name: Name of the model

    Returns:
        Path to the model directory
    """
    return MODELS_DIR / model_name


def get_dataset_dir(dataset_name: str) -> Path:
    """Get the directory path for a specific dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Path to the dataset directory
    """
    return DATASETS_DIR / dataset_name


def get_prompt_dir(prompt_name: str) -> Path:
    """Get the directory path for a specific prompt.

    Args:
        prompt_name: Name of the prompt

    Returns:
        Path to the prompt directory
    """
    return PROMPTS_DIR / prompt_name
