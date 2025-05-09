from abc import ABC, abstractmethod
from openagentkit.core.models.responses.embedding_response import EmbeddingResponse, EmbeddingUnit
from typing import Union, Literal

class AsyncBaseEmbeddingModel(ABC):
    """
    An abstract base class for async embedding models.

    ## Methods:
        `encode_texts()`: An abstract method to encode texts into embeddings.

        `tokenize_texts()`: An abstract method to tokenize texts.
    """
    @abstractmethod
    async def encode_query(self, query: str, include_metadata: bool = False) -> Union[list[EmbeddingUnit], EmbeddingResponse]:
        """
        An abstract method to encode a query into an embedding.

        Args:
            query (str): The query to encode.
            include_metadata (bool): Whether to include metadata in the response.

        Returns:
            Union[list[EmbeddingUnit], EmbeddingResponse]: The embeddings response. 
            If `include_metadata` is `True`, return an `EmbeddingResponse` object containing the embedding. 
            If `include_metadata` is `False`, return a list of `EmbeddingUnit` objects containing the embedding.
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
    def tokenize_texts(self, texts: list[str]) -> list[list[int]]:
        """
        An abstract method to tokenize texts.

        Args:
            texts (list[str]): The texts to tokenize.

        Returns:
            list[list[int]]: The tokenized texts.
        """
        raise NotImplementedError("tokenize_texts method must be implemented")