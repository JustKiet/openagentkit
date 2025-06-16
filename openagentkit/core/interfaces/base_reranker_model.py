from abc import ABC, abstractmethod
from typing import Union
from openagentkit.core.models.responses.reranking_response import RerankingResponse
from openagentkit.core.models.io.reranking import RerankingUnit

class BaseRerankerModel(ABC):
    @abstractmethod
    def rerank(self, 
               query: str, 
               items: list[str],
               top_k: int,
               include_metadata: bool = True) -> Union[list[RerankingUnit], RerankingResponse]:
        """
        An abstract method to rerank a list of items based on a query.

        Args:
            query (str): The query to use for reranking.
            items (list[str]): The list of items to rerank.
            top_k (int): The number of top items to return.

        Returns:
            if `include_metadata` is `True`, return an `RerankingResponse` object containing the reranked items with metadata.
            if `include_metadata` is `False`, return a list of `RerankingUnit` objects containing the reranked items.
        """
        raise NotImplementedError("rerank method must be implemented")