from pydantic import BaseModel, Field


class ModelMetadata(BaseModel):
    name: str
    hf_path: str
    model_type: str = "causal"
    chat_template: str
    load_in_4bit: bool = True


class DatasetMetadata(BaseModel):
    name: str
    format: str
    train_path: str
    split_size: int = 1000


class TrainingConfig(BaseModel):
    base_model: str
    dataset_name: str
    output_dir: str
    epochs: int = 1
    learning_rate: float = 2e-4
