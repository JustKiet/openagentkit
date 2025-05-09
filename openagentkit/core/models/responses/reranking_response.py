from pydantic import BaseModel
from openagentkit.core.models.io.reranking import RerankingUnit

class RerankingResponse(BaseModel):
    """
    A fully populated reranking response.

    Schema:
        ```python
        class RerankingResponse(BaseModel):
            query: str
            results: list[RerankingUnit]
            reranking_model: str
            total_tokens: int
        ```
    Where:
        - `query`: The query used for reranking.
        - `results`: A list of reranking units.
        - `reranking_model`: The reranking model used.
        - `total_tokens`: The total tokens used.
    """
    query: str
    results: list[RerankingUnit]
    reranking_model: str
    total_tokens: int