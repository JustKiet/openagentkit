from abc import ABC, abstractmethod
from openagentkit.models.responses.embedding_response import EmbeddingResponse

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def encode_texts(self, texts: list[str]) -> EmbeddingResponse:
        raise NotImplementedError("encode_texts method must be implemented")
    
    @abstractmethod
    def tokenize_texts(self, texts: list[str]) -> list[list[int]]:
        raise NotImplementedError("tokenize_texts method must be implemented")