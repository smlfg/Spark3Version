from pydantic import BaseModel, Field
from typing import List, Optional

class ModelMetadata(BaseModel):
    name: str
    hf_path: str
    model_type: str = "causal"
    chat_template: str = "chatml"
    load_in_4bit: bool = True
    target_modules: List[str] = ["q_proj", "v_proj"]

class DatasetMetadata(BaseModel):
    name: str
    format: str = "jsonl"
    train_path: str
    split_size: int = 1000

class TrainingConfig(BaseModel):
    base_model: str
    dataset_name: str
    output_dir: str
    epochs: int = 1
    learning_rate: float = 2e-4
