from abc import ABC, abstractmethod
from openagentkit.core.models.responses.embedding_response import EmbeddingResponse, EmbeddingUnit
from typing import Union, TypeVar, Generic

T = TypeVar("T", int, str)

class AsyncBaseEmbeddingModel(ABC, Generic[T]):
    @abstractmethod
    async def encode_query(self, query: str, include_metadata: bool = False) -> Union[EmbeddingUnit, EmbeddingResponse]:
        """
        An abstract method to encode a query into an embedding.

        Args:
            query (str): The query to encode.
            include_metadata (bool): Whether to include metadata in the response.

        Returns:
            Union[EmbeddingUnit, EmbeddingResponse]: The embeddings response. 
            If `include_metadata` is `True`, return an `EmbeddingResponse` object containing the query embedding response. 
            If `include_metadata` is `False`, return an `EmbeddingUnit` object containing the query embedding.
        """
        raise NotImplementedError("encode_query method must be implemented")

    @abstractmethod
    async def encode_texts(self, texts: list[str], include_metadata: bool = False) -> Union[list[EmbeddingUnit], EmbeddingResponse]:
        """
        An abstract method to encode texts into embeddings.

        Args:
            texts (list[str]): The texts to encode.
            include_metadata (bool): Whether to include metadata in the response.

        Returns:
            Union[list[EmbeddingUnit], EmbeddingResponse]: The embeddings response. 
            If `include_metadata` is `True`, return an `EmbeddingResponse` object containing the embeddings. 
            If `include_metadata` is `False`, return a list of `EmbeddingUnit` objects containing the embeddings.
        """
        raise NotImplementedError("encode_texts method must be implemented")
    
    @abstractmethod
    def tokenize_texts(self, texts: list[str]) -> list[list[T]]:
        """
        An abstract method to tokenize texts.

        Args:
            texts (list[str]): The texts to tokenize.

        Returns:
            list[list[int]]: The tokenized texts.
        """
        raise NotImplementedError("tokenize_texts method must be implemented")