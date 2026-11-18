from pydantic import BaseModel, Field
from typing import Literal


class DataConfig(BaseModel):
    path: str = "data/raw/iris.csv"
    chunk_size: int = Field(default=50, gt=0)
    random_state: int = 42


class ModelConfig(BaseModel):
    C: float = Field(default=1.0, gt=0)
    max_iter: int = Field(default=200, gt=0)
    mode: Literal["full", "chunks"] = "full"


class Lab2Config(BaseModel):
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    target_col: str = "target"
