from openagentkit.core.interfaces import BaseRerankerModel
from openagentkit.core.models.responses.reranking_response import RerankingResponse
from openagentkit.core.models.io.reranking import RerankingUnit
from voyageai import AsyncClient
from typing import Literal, Union
import os

class AsyncVoyageRerankerModel(BaseRerankerModel):
    def __init__(self, 
                 client: AsyncClient = None,
                 api_key: str = os.getenv("VOYAGE_API_KEY"),
                 reranking_model: Literal["rerank-2", "rerank-2-lite"] = "rerank-2",
                 ):
        """
        Initialize the VoyageRerankerModel with a Voyage client.

        Args:
            client (Client): The Voyage client to use for reranking.
        """
        self._client = client
        if self._client is None:
            if api_key is None:
                raise ValueError("No API key provided. Please set the VOYAGE_API_KEY environment variable or pass it as an argument.")
            self._client = AsyncClient(api_key=api_key)

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

        reranking_units = []

        for item in response.results:
            reranking_units.append(
                RerankingUnit(
                    index=item.index,
                    content=item.document,
                    score=item.score,
                )
            )

        if include_metadata:
            return RerankingResponse(
                query=query,
                reranking_units=reranking_units,
                model=self._reranking_model,
                metadata=response.metadata
            )
        else:
            return reranking_units
        