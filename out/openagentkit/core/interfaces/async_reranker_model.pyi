import abc
from abc import ABC, abstractmethod
from openagentkit.core.models.io.reranking import RerankingUnit as RerankingUnit
from openagentkit.core.models.responses.reranking_response import RerankingResponse as RerankingResponse

class AsyncBaseRerankerModel(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    async def rerank(self, query: str, items: list[str], top_k: int, include_metadata: bool = False) -> list[RerankingUnit] | RerankingResponse: ...
