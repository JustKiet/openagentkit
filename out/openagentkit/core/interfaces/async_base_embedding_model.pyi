import abc
from abc import ABC, abstractmethod
from openagentkit.core.models.responses.embedding_response import EmbeddingResponse as EmbeddingResponse, EmbeddingUnit as EmbeddingUnit
from typing import Generic, TypeVar

T = TypeVar('T', int, str)

class AsyncBaseEmbeddingModel(ABC, Generic[T], metaclass=abc.ABCMeta):
    @abstractmethod
    async def encode_query(self, query: str, include_metadata: bool = False) -> EmbeddingUnit | EmbeddingResponse: ...
    @abstractmethod
    async def encode_texts(self, texts: list[str], include_metadata: bool = False) -> list[EmbeddingUnit] | EmbeddingResponse: ...
    @abstractmethod
    def tokenize_texts(self, texts: list[str]) -> list[list[T]]: ...
