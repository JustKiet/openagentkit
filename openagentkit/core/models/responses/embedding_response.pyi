from openagentkit.core.models.io.embeddings import EmbeddingUnit as EmbeddingUnit
from pydantic import BaseModel

class EmbeddingResponse(BaseModel):
    embeddings: list[EmbeddingUnit]
    embedding_model: str
    total_tokens: int
