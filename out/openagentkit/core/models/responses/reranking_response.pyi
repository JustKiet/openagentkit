from openagentkit.core.models.io.reranking import RerankingUnit as RerankingUnit
from pydantic import BaseModel

class RerankingResponse(BaseModel):
    query: str
    results: list[RerankingUnit]
    reranking_model: str
    total_tokens: int
