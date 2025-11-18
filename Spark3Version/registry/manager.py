import yaml
from pathlib import Path
from typing import List
from utils.paths import MODELS_DIR, DATASETS_DIR
from registry.schemas import ModelMetadata, DatasetMetadata

class RegistryManager:
    @staticmethod
    def list_models() -> List[str]:
        if not MODELS_DIR.exists(): return []
        return [d.name for d in MODELS_DIR.iterdir() if d.is_dir() and (d / "config.yaml").exists()]

    @staticmethod
    def get_model(name: str) -> ModelMetadata:
        config_path = MODELS_DIR / name / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Model {name} not found at {config_path}")
        
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        return ModelMetadata(**data)

    @staticmethod
    def list_datasets() -> List[str]:
        if not DATASETS_DIR.exists(): return []
        return [d.name for d in DATASETS_DIR.iterdir() if d.is_dir() and (d / "metadata.yaml").exists()]

    @staticmethod
    def get_dataset(name: str) -> DatasetMetadata:
        meta_path = DATASETS_DIR / name / "metadata.yaml"
        if not meta_path.exists():
            raise FileNotFoundError(f"Dataset {name} not found at {meta_path}")
            
        with open(meta_path, "r") as f:
            data = yaml.safe_load(f)
        return DatasetMetadata(**data)
