from openagentkit.core.interfaces import AsyncBaseRerankerModel
from openagentkit.core.models.responses.reranking_response import RerankingResponse
from openagentkit.core.models.io.reranking import RerankingUnit
from voyageai.client_async import AsyncClient
from typing import Literal, Union, Optional, overload
import os

class AsyncVoyageAIRerankerModel(AsyncBaseRerankerModel):
    def __init__(self, 
                 client: Optional[AsyncClient] = None,
                 api_key: Optional[str] = os.getenv("VOYAGE_API_KEY"),
                 reranking_model: Literal["rerank-2", "rerank-2-lite"] = "rerank-2",
                 ):
        
        if client is None:
            if api_key is None:
                raise ValueError("No API key provided. Please set the VOYAGE_API_KEY environment variable or pass it as an argument.")
            self._client = AsyncClient(api_key=api_key)
        else:
            self._client = client

        self._reranking_model = reranking_model

    @property
    def reranking_model(self) -> str:
        """
        Get the reranking model.

        Returns:
            The reranking model.
        """
        return self._reranking_model
    
    @reranking_model.setter
    def reranking_model(self, value: str) -> None:
        """
        Set the reranking model.

        Args:
            value: The reranking model to set.
        """
        if value not in ["rerank-2", "rerank-2-lite"]:
            raise ValueError("Invalid reranking model. Must be 'rerank-2' or 'rerank-2-lite'.")
        self._reranking_model = value

    @overload
    async def rerank(self,
                     query: str, 
                     items: list[str],
                     top_k: int,
                     include_metadata: Literal[True]) -> RerankingResponse:
        ...
    
    @overload
    async def rerank(self,
                     query: str, 
                     items: list[str],
                     top_k: int,
                     include_metadata: Literal[False]) -> list[RerankingUnit]:
        ...

    async def rerank(self,
                     query: str, 
                     items: list[str],
                     top_k: int,
                     include_metadata: bool = True) -> Union[list[RerankingUnit], RerankingResponse]:
        """
        Rerank a list of items based on a query.

        Args:
              query (str): The query to use for reranking.
              items (list[str]): The list of items to rerank.
              top_k (int): The number of top items to return.
              include_metadata (bool): Whether to include metadata in the response.

        Returns:
              If `include_metadata` is `True`, return an `RerankingResponse` object containing the reranked items with metadata.
              If `include_metadata` is `False`, return a list of `RerankingUnit` objects containing the reranked items.
        """
        response = await self._client.rerank(
              model=self._reranking_model,
              query=query,
              documents=items,
              top_k=top_k
        )

        reranking_units: list[RerankingUnit] = []

        for item in response.results:
            reranking_units.append(
                RerankingUnit(
                    index=item.index, # type: ignore
                    content=item.document, # type: ignore
                    relevance_score=item.relevance_score, # type: ignore
                )
            )

        if include_metadata:
            return RerankingResponse(
                query=query,
                results=reranking_units,
                reranking_model=self._reranking_model,
                total_tokens=response.total_tokens,   
            )
        else:
            return reranking_units
        