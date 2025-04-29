import tiktoken
from openai import OpenAI
from openai.types import CreateEmbeddingResponse
from openagentkit.core.interfaces.base_embedding_model import BaseEmbeddingModel
from openagentkit.core.models.io.embeddings import EmbeddingUnit
from openagentkit.core.models.responses import EmbeddingResponse
from typing import Literal, Union

class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, 
                 client: OpenAI,
                 embedding_model: Literal[
                     "text-embedding-3-small", 
                     "text-embedding-3-large", 
                     "text-embedding-ada-002",
                 ] = "text-embedding-3-small",
                 embedding_encoding: str = "cl100k_base",
                 encoding_format: Literal["float", "base64"] = "float"):
        self.client = client
        self.embedding_model = embedding_model
        self.embedding_encoding = embedding_encoding
        self.encoding_format = encoding_format

        match self.embedding_model:
            case "text-embedding-3-small":
                self.dimensions = 1536
            case "text-embedding-3-large":
                self.dimensions = 3072
            case "text-embedding-ada-002":
                self.dimensions = 1536

    def encode_query(self, 
                     query: str,
                     include_metadata: bool = False) -> Union[EmbeddingUnit, EmbeddingResponse]:
        """
        Encode a query into an embedding.

        Args:
            query: A single query to encode.
            include_metadata: Whether to include metadata in the response. (default: `False`)
        Returns:
            If `include_metadata` is `True`, return an `EmbeddingResponse` object containing the embedding.
            
            If `include_metadata` is `False`, return an `EmbeddingUnit` object containing the embedding.
        Schema:
            ```python
            class EmbeddingResponse(BaseModel):
                embeddings: list[EmbeddingUnit] # List of embeddings
                embedding_model: str # The embedding model used
                total_tokens: int # The total number of tokens used

            class EmbeddingUnit(BaseModel):
                index: int # The index of the embedding
                object: str # The object of the embedding
                embedding: list[float] # The embedding vector
            ```
        Example:
            ```python
            from openagentkit.modules.openai import OpenAIEmbeddingModel

            embedding_model = OpenAIEmbeddingModel()
            embedding_response = embedding_model.encode_query(
                query="Hello, world!", 
                include_metadata=True
            )
            # Get the embedding
            embedding: list[float] = embedding_response.embeddings[0].embedding
            # Get the usage
            total_tokens: int = embedding_response.total_tokens
            # Get the embedding model
            embedding_model: str = embedding_response.embedding_model
            ```
        """
        embedding_response: EmbeddingResponse = self.encode_texts(
            texts=[query],
            include_metadata=True
        )

        if include_metadata:
            return embedding_response
        else:
            return embedding_response.embeddings[0]

    def encode_texts(self, 
                     texts: list[str],
                     include_metadata: bool = False) -> Union[list[EmbeddingUnit], EmbeddingResponse]:
        """
        Encode a list of texts into a list of embeddings.
        Args:
            texts: A list of texts to encode.
            include_metadata: Whether to include metadata in the response. (default: `False`)
        Returns:
            If `include_metadata` is `True`, return an `EmbeddingResponse` object containing the embeddings.
            If `include_metadata` is `False`, return a list of `EmbeddingUnit` objects containing the embeddings.
        Schema:
            ```python
            class EmbeddingResponse(BaseModel):
                embeddings: list[EmbeddingUnit] # List of embeddings
                embedding_model: str # The embedding model used
                total_tokens: int # The total number of tokens used

            class EmbeddingUnit(BaseModel):
                index: int # The index of the embedding
                object: str # The object of the embedding
                embedding: list[float] # The embedding vector
            ```
        Example:
            ```python
            from openagentkit.modules.openai import OpenAIEmbeddingModel
            
            embedding_model = OpenAIEmbeddingModel()
            embedding_response = embedding_model.encode_texts(
                texts=["Hello, world!", "This is a test."],
                include_metadata=True
            )
            # Get the embeddings
            embeddings: list[EmbeddingUnit] = embedding_response.embeddings
            # Get the usage
            total_tokens: int = embedding_response.total_tokens
            # Get the embedding model
            embedding_model: str = embedding_response.embedding_model
            ```
        """
        formatted_texts: list[str] = []
        for text in texts:
            text = text.replace("\n", " ")
            formatted_texts.append(text)

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=formatted_texts,
            encoding_format=self.encoding_format,
        )

        embeddings: list[EmbeddingUnit] = []
        
        for embedding in response.data:
            embeddings.append(
                EmbeddingUnit(
                    index=embedding.index,
                    object=embedding.object,
                    content=formatted_texts[embedding.index],
                    embedding=embedding.embedding,
                    type=self.encoding_format
                )
            )

        if include_metadata:
            return EmbeddingResponse(
                embeddings=embeddings,
                embedding_model=self.embedding_model,
                total_tokens=response.usage.total_tokens,
            )
        else:
            return embeddings
    
    def tokenize_texts(self, texts: list[str]) -> list[list[int]]:
        """
        Tokenize a list of texts into a list of tokens.
        Args:
            texts: A list of texts to tokenize.
        Returns:
            A list of tokens lists for each text.
        Example:
            ```python
            from openagentkit.modules.openai import OpenAIEmbeddingModel
            
            embedding_model = OpenAIEmbeddingModel()
            tokens = embedding_model.tokenize_texts(
                texts=["Hello, world!", "This is a test."]
            )
            print(tokens) >>> [[9906, 11, 1917, 0], [2028, 374, 264, 1296, 13]]
            ```
        """
        encoder = tiktoken.get_encoding(self.embedding_encoding)

        tokens: list[int] = []

        for text in texts:
            tokens.append(encoder.encode(text))

        return tokens