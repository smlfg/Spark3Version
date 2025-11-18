from pydantic import BaseModel, Field
from typing import List, Optional


class ModelMetadata(BaseModel):
    name: str
    hf_path: str
    model_type: str = "causal"
    chat_template: str
    target_modules: List[str]


class DatasetMetadata(BaseModel):
    name: str
    format: str
    train_path: str
    system_column: Optional[str] = None
    input_column: str
    output_column: str


class TrainingConfig(BaseModel):
    base_model: str
    dataset_name: str
    output_dir: str
    epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    use_4bit: bool = True
