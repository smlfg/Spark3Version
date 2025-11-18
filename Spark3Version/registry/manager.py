"""Registry Manager for Models and Datasets."""
from pathlib import Path
from typing import List, Dict
import json
from Spark3Version.registry.schemas import ModelMetadata, DatasetMetadata
from Spark3Version.utils.paths import MODELS_DIR, DATASETS_DIR


class RegistryManager:
    """Manages registration and discovery of models and datasets."""

    def __init__(self):
        self.models_dir = MODELS_DIR
        self.datasets_dir = DATASETS_DIR

    def get_available_models(self) -> List[str]:
        """Scan and return list of registered models."""
        models = []

        # Check for JSON registry files
        registry_file = self.models_dir / "registry.json"
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                data = json.load(f)
                models.extend([m['name'] for m in data.get('models', [])])

        # Also scan for directories
        if self.models_dir.exists():
            for item in self.models_dir.iterdir():
                if item.is_dir() and item.name not in models and not item.name.startswith('.'):
                    models.append(item.name)

        return sorted(models) if models else ["No models registered"]

    def get_available_datasets(self) -> List[str]:
        """Scan and return list of registered datasets."""
        datasets = []

        # Check for JSON registry files
        registry_file = self.datasets_dir / "registry.json"
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                data = json.load(f)
                datasets.extend([d['name'] for d in data.get('datasets', [])])

        # Also scan for JSONL files
        if self.datasets_dir.exists():
            for item in self.datasets_dir.iterdir():
                if item.suffix == '.jsonl' and item.stem not in datasets:
                    datasets.append(item.stem)

        return sorted(datasets) if datasets else ["No datasets registered"]

    def get_model_metadata(self, name: str) -> Dict:
        """Get metadata for a specific model."""
        registry_file = self.models_dir / "registry.json"
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                data = json.load(f)
                for model in data.get('models', []):
                    if model['name'] == name:
                        return model
        return {}

    def get_dataset_metadata(self, name: str) -> Dict:
        """Get metadata for a specific dataset."""
        registry_file = self.datasets_dir / "registry.json"
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                data = json.load(f)
                for dataset in data.get('datasets', []):
                    if dataset['name'] == name:
                        return dataset
        return {}
