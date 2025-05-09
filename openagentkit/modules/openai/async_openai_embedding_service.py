import tiktoken
from openai import AsyncOpenAI
from openagentkit.core.interfaces.async_base_embedding_model import AsyncBaseEmbeddingModel
from openagentkit.core.models.io.embeddings import EmbeddingUnit
from openagentkit.core.models.responses import EmbeddingResponse
from typing import Literal, Union
import os

class AsyncOpenAIEmbeddingModel(AsyncBaseEmbeddingModel):
    def __init__(self, 
                 client: AsyncOpenAI = None,
                 api_key: str = os.getenv("OPENAI_API_KEY"),
                 embedding_model: Literal[
                     "text-embedding-3-small", 
                     "text-embedding-3-large", 
                     "text-embedding-ada-002",
                 ] = "text-embedding-3-small",
                 embedding_encoding: str = "cl100k_base",
                 encoding_format: Literal["float", "base64"] = "float"):
        super().__init__(encoding_format=encoding_format)
        self._client = client
        if self.client is None:
            if api_key is None:
                raise ValueError("No API key provided. Please set the OPENAI_API_KEY environment variable or pass it as an argument.")
            self.client = AsyncOpenAI(api_key=api_key)

        self._embedding_model = embedding_model
        self._embedding_encoding = embedding_encoding

        match self.embedding_model:
            case "text-embedding-3-small":
                self._dimensions = 1536
            case "text-embedding-3-large":
                self._dimensions = 3072
            case "text-embedding-ada-002":
                self._dimensions = 1536

        self._encoding_format = encoding_format

    @property
    def encoding_format(self) -> str:
        """
        Get the encoding format.
        Returns:
            The encoding format.
        """
        return self._encoding_format
    
    @encoding_format.setter
    def encoding_format(self, value: str) -> None:
        """
        Set the encoding format.
        Args:
            value: The encoding format to set. Can be "float" or "base64".
        """
        if value not in ["float", "base64"]:
            raise ValueError("Invalid encoding format. Must be 'float' or 'base64'.")
        self._encoding_format = value
        return self._encoding_format
    
    @property
    def embedding_model(self) -> str:
        """
        Get the embedding model.
        Returns:
            The embedding model.
        """
        return self._embedding_model
    
    @embedding_model.setter
    def embedding_model(self, value: str) -> None:
        """
        Set the embedding model.
        Args:
            value: The embedding model to set.
        """
        if value not in ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]:
            raise ValueError("Invalid embedding model. Must be 'text-embedding-3-small', 'text-embedding-3-large', or 'text-embedding-ada-002'.")
        self._embedding_model = value
    
    @property
    def embedding_encoding(self) -> str:
        """
        Get the embedding encoding.
        Returns:
            The embedding encoding.
        """
        return self._embedding_encoding
    
    @embedding_encoding.setter
    def embedding_encoding(self, value: str) -> None:
        """
        Set the embedding encoding.
        Args:
            value: The embedding encoding to set. Currently, only "cl100k_base" is supported.
        """
        if value != "cl100k_base":
            raise ValueError("Invalid embedding encoding. Currently, only 'cl100k_base' is supported.")
        self._embedding_encoding = value
    
    @property
    def encoding_format(self) -> str:
        """
        Get the encoding format.
        Returns:
            The encoding format.
        """
        return self._encoding_format
    
    @encoding_format.setter
    def encoding_format(self, value: str) -> None:
        """
        Set the encoding format.
        Args:
            value: The encoding format to set. Can be "float" or "base64".
        """
        if value not in ["float", "base64"]:
            raise ValueError("Invalid encoding format. Must be 'float' or 'base64'.")
        self._encoding_format = value

    async def encode_query(self, 
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
        embedding_response: EmbeddingResponse = await self.encode_texts(
            texts=[query],
            include_metadata=True
        )

        if include_metadata:
            return embedding_response
        else:
            return embedding_response.embeddings[0]

    async def encode_texts(self, 
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

        response = await self.client.embeddings.create(
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