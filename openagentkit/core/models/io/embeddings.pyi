from pydantic import BaseModel
from typing import Literal

class EmbeddingSplits(BaseModel):
    content: str
    index: int
    combined_splits: str | None

class EmbeddingUnit(BaseModel):
    index: int
    object: str
    content: str
    embedding: list[float] | list[int] | str
    type: Literal['base64', 'float']
