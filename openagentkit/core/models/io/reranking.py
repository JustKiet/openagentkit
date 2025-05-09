from pydantic import BaseModel

class RerankingUnit(BaseModel):
    """
    A reranking unit.

    Schema:
        ```python
        class RerankingUnit(BaseModel):
            index: int
            content: str
            score: float
        ```
    Where:
        - `index`: The index of the reranking unit.
        - `content`: The content of the reranking unit.
        - `score`: The score of the reranking unit.
    """
    index: int
    content: str
    relevance_score: float