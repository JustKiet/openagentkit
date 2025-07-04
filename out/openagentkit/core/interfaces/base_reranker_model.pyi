import abc
from abc import ABC, abstractmethod
from openagentkit.core.models.io.reranking import RerankingUnit as RerankingUnit
from openagentkit.core.models.responses.reranking_response import RerankingResponse as RerankingResponse

class BaseRerankerModel(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def rerank(self, query: str, items: list[str], top_k: int, include_metadata: bool = True) -> list[RerankingUnit] | RerankingResponse: ...
