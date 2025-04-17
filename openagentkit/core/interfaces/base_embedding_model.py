from abc import ABC, abstractmethod
from openagentkit.core.models.responses.embedding_response import EmbeddingResponse

class BaseEmbeddingModel(ABC):
    """
    An abstract base class for embedding models.

    ## Methods:
        `encode_texts()`: An abstract method to encode texts into embeddings.

        `tokenize_texts()`: An abstract method to tokenize texts.
    """
    @abstractmethod
    def encode_texts(self, texts: list[str]) -> EmbeddingResponse:
        """
        An abstract method to encode texts into embeddings.

        Args:
            texts (list[str]): The texts to encode.

        Returns:
            EmbeddingResponse: The embeddings response.
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