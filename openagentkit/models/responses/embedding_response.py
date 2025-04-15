from pydantic import BaseModel
from openagentkit.models.responses.usage_responses import EmbeddingUsageResponse

class EmbeddingUnit(BaseModel):
    """
    An embedding unit.

    Schema:
        ```python
        class EmbeddingUnit(BaseModel):
            index: int
            object: str
            embedding: list[float]
        ```
    Where:
        - `index`: The index of the embedding.
        - `object`: The object of the embedding.
        - `embedding`: The embedding vector.
    """
    index: int
    object: str
    embedding: list[float]

class EmbeddingResponse(BaseModel):
    """
    An embedding response.

    Schema:
        ```python
        class EmbeddingResponse(BaseModel):
            embeddings: list[EmbeddingUnit]
            embedding_model: str
            usage: EmbeddingUsageResponse
        ```
    Where:
        - `embeddings`: A list of embedding units.
        - `embedding_model`: The embedding model used.
        - `usage`: The usage of the embedding model.
    """
    embeddings: list[EmbeddingUnit]
    embedding_model: str