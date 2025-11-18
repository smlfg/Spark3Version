"""
Data schemas for registry metadata.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ModelMetadata(BaseModel):
    """Metadata for a registered model."""
    name: str
    version: str
    description: Optional[str] = None
    base_model: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    tags: Optional[List[str]] = Field(default_factory=list)


class DatasetMetadata(BaseModel):
    """Metadata for a registered dataset."""
    name: str
    version: str
    description: Optional[str] = None
    format: Optional[str] = None
    size: Optional[int] = None
    num_samples: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    tags: Optional[List[str]] = Field(default_factory=list)
    columns: Optional[List[str]] = Field(default_factory=list)
