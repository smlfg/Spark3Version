"""
Registry Manager for managing models and datasets.
"""
from typing import List
from pathlib import Path
import yaml

from utils.paths import MODELS_DIR, DATASETS_DIR, PROMPTS_DIR
from registry.schemas import ModelMetadata, DatasetMetadata


class RegistryManager:
    """Manager for registry operations."""

    def list_models(self) -> List[str]:
        """
        Scan MODELS_DIR for folders containing config.yaml.

        Returns:
            List of model names (folder names).
        """
        model_names = []

        if not MODELS_DIR.exists():
            return model_names

        for item in MODELS_DIR.iterdir():
            if item.is_dir():
                config_file = item / "config.yaml"
                if config_file.exists():
                    model_names.append(item.name)

        return sorted(model_names)

    def get_model(self, name: str) -> ModelMetadata:
        """
        Load model metadata from config.yaml.

        Args:
            name: Name of the model.

        Returns:
            ModelMetadata object.

        Raises:
            FileNotFoundError: If model does not exist.
        """
        model_dir = MODELS_DIR / name
        config_file = model_dir / "config.yaml"

        if not model_dir.exists():
            raise FileNotFoundError(f"Model '{name}' does not exist")

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found for model '{name}'")

        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        return ModelMetadata(**config_data)

    def list_datasets(self) -> List[str]:
        """
        Scan DATASETS_DIR for folders containing metadata.yaml.

        Returns:
            List of dataset names (folder names).
        """
        dataset_names = []

        if not DATASETS_DIR.exists():
            return dataset_names

        for item in DATASETS_DIR.iterdir():
            if item.is_dir():
                metadata_file = item / "metadata.yaml"
                if metadata_file.exists():
                    dataset_names.append(item.name)

        return sorted(dataset_names)

    def get_dataset(self, name: str) -> DatasetMetadata:
        """
        Load dataset metadata from metadata.yaml.

        Args:
            name: Name of the dataset.

        Returns:
            DatasetMetadata object.

        Raises:
            FileNotFoundError: If dataset does not exist.
        """
        dataset_dir = DATASETS_DIR / name
        metadata_file = dataset_dir / "metadata.yaml"

        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset '{name}' does not exist")

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found for dataset '{name}'")

        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata_data = yaml.safe_load(f)

        return DatasetMetadata(**metadata_data)
