from pydantic import BaseModel

class RerankingUnit(BaseModel):
    index: int
    content: str
    relevance_score: float
