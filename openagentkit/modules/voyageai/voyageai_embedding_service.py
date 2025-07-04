from openagentkit.core.interfaces import BaseEmbeddingModel
from openagentkit.core.models.io.embeddings import EmbeddingUnit
from openagentkit.core.models.responses import EmbeddingResponse
from typing import Literal, Union, Optional, overload
from voyageai.client import Client
import base64
import os
import warnings
import struct

class VoyageAIEmbeddingModel(BaseEmbeddingModel[str]):
    def __init__(self,
                 client: Optional[Client] = None,
                 api_key: Optional[str] = os.getenv("VOYAGE_API_KEY"),
                 embedding_model: Literal[
                     "voyage-3-large",
                     "voyage-3",
                     "voyage-3-lite",
                     "voyage-code-3",
                     "voyage-finance-2",
                     "voyage-law-2",
                     "voyage-code-2",
                 ] = "voyage-3-large",
                 dimensions: Literal[256, 512, 1024, 1536, 2048] = 1024,
                 encoding_format: Literal["float", "base64"] = "float"):
        
        if client is None:
            if api_key is None:
                raise ValueError("No API key provided. Please set the VOYAGE_API_KEY environment variable or pass it as an argument.")
            self._client = Client(api_key=api_key)
        else:
            self._client = client
        
        self._embedding_model = embedding_model
        self._encoding_format: Literal["float", "base64"] = encoding_format
        self._dimensions = self._handle_model_dimensions(embedding_model, dimensions)

    @property
    def encoding_format(self) -> str:
        """
        Get the encoding format.
        Returns:
            The encoding format.
        """
        return self._encoding_format
    
    @encoding_format.setter
    def encoding_format(self, value: Literal["float", "base64"]) -> None:
        """
        Set the encoding format.
        Args:
            value: The encoding format to set. Can be "float" or "base64".
        """
        if value not in ["float", "base64"]:
            raise ValueError("Invalid encoding format. Supported formats are 'float' and 'base64'.")
        self._encoding_format = value

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
            value: The embedding model to set. Supported models are:
                - "voyage-3-large"
                - "voyage-3"
                - "voyage-3-lite"
                - "voyage-code-3"
                - "voyage-finance-2"
                - "voyage-law-2"
                - "voyage-code-2"
        Returns:
            The embedding model.
        """
        if value not in [
            "voyage-3-large",
            "voyage-3",
            "voyage-3-lite",
            "voyage-code-3",
            "voyage-finance-2",
            "voyage-law-2",
            "voyage-code-2",
        ]:
            raise ValueError("Invalid embedding model. Supported models are: 'voyage-3-large', 'voyage-3', 'voyage-3-lite', 'voyage-code-3', 'voyage-finance-2', 'voyage-law-2', 'voyage-code-2'.")
        self._embedding_model = value
    
    @property
    def dimensions(self) -> int:
        """
        Get the dimensions of the embedding model.
        Returns:
            The dimensions of the embedding model.
        """
        return self._dimensions
    
    @dimensions.setter
    def dimensions(self, value: Literal[256, 512, 1024, 1536, 2048]) -> None:
        """
        Set the dimensions of the embedding model.
        Args:
            value: The dimensions to set. Supported dimensions are:
                - 256
                - 512
                - 1024
                - 1536
                - 2048
        Returns:
            The dimensions of the embedding model.
        """
        if value not in [256, 512, 1024, 1536, 2048]:
            raise ValueError("Invalid dimensions. Supported dimensions are: 256, 512, 1024, 1536, 2048.")
        self._dimensions = self._handle_model_dimensions(self.embedding_model, value)
    
    def _handle_model_dimensions(self, 
                                 embedding_model: str,
                                 dimensions: Literal[256, 512, 1024, 1536, 2048]) -> int:
        match embedding_model:
            case "voyage-3-large":
                if dimensions not in [256, 512, 1024, 2048]:
                    warnings.warn("voyage-3-large model is only available with 256, 512, 1024, or 2048 dimensions. Setting dimensions to 1024.")
                    return 1024
                else:
                    return dimensions
            case "voyage-3":
                if dimensions != 1024:
                    warnings.warn("voyage-3 model is only available with 1024 dimensions. Setting dimensions to 1024.")
                    return 1024
                else:
                    return dimensions
            case "voyage-3-lite":
                if dimensions != 512:
                    warnings.warn("voyage-3-lite model is only available with 512 dimensions. Setting dimensions to 512.")
                    return 512
                else:
                    return dimensions
            case "voyage-code-3":
                if dimensions not in [256, 512, 1024, 2048]:
                    warnings.warn("voyage-3-large model is only available with 256, 512, 1024, or 2048 dimensions. Setting dimensions to 1024.")
                    return 1024
                else:
                    return dimensions
            case "voyage-finance-2":
                if dimensions != 1024:
                    warnings.warn("voyage-finance-2 model is only available with 1024 dimensions. Setting dimensions to 2048.")
                    return 1024
                else:
                    return dimensions
            case "voyage-law-2":
                if dimensions != 1024:
                    warnings.warn("voyage-finance-2 model is only available with 1024 dimensions. Setting dimensions to 2048.")
                    return 1024
                else:
                    return dimensions
            case "voyage-code-2":
                if dimensions != 1536:
                    warnings.warn("voyage-code-2 model is only available with 1536 dimensions. Setting dimensions to 1536.")
                    return 1536
                else:
                    return dimensions
            case _:
                raise ValueError("Invalid embedding model. Supported models are: 'voyage-3-large', 'voyage-3', 'voyage-3-lite', 'voyage-code-3', 'voyage-finance-2', 'voyage-law-2', 'voyage-code-2'.")
    
    @overload
    def encode_query(self,
                           query: str,
                           include_metadata: Literal[True],
                           truncation: bool = True) -> EmbeddingResponse:
        ...

    @overload
    def encode_query(self,
                           query: str,
                           include_metadata: Literal[False],
                           truncation: bool = True) -> EmbeddingUnit:
        ...

    def encode_query(self, 
                     query: str, 
                     include_metadata: bool = False,
                     truncation: bool = True,) -> Union[EmbeddingUnit, EmbeddingResponse]:
        """
        Encode a query into an embedding.

        Args:
            query (str): The query to encode.
            include_metadata (bool): Whether to include metadata in the response.

        Returns:
            Returns:
            If `include_metadata` is `True`, return an `EmbeddingResponse` object containing the embedding with metadata.
            
            If `include_metadata` is `False`, return an `EmbeddingUnit` object containing the embedding.
        """
        embedding_response: EmbeddingResponse = self.encode_texts(
            texts=[query],
            input_type="query",
            truncation=truncation,
            include_metadata=True
        )

        if include_metadata:
            return embedding_response
        else:
            return embedding_response.embeddings[0]
        
    def _handle_base64_encoding(
        self, 
        embedding: list[float] | list[int],
    ) -> str:
        """
        Handle base64 encoding of the embedding.
        
        Args:
            embedding (bytes): The embedding to encode.
        
        Returns:
            str: The base64 encoded string.
        """
        embedding = struct.pack(f"{len(embedding)}f", *embedding) # type: ignore
        return base64.b64encode(embedding).decode("utf-8") # type: ignore

    @overload
    def encode_texts(self,
                           texts: list[str],
                           include_metadata: Literal[True],
                           input_type: Optional[Literal["query", "document"]] = "document",
                           truncation: bool = True) -> EmbeddingResponse:
        ...

    @overload
    def encode_texts(self,
                           texts: list[str],
                           include_metadata: Literal[False],
                           input_type: Optional[Literal["query", "document"]] = "document",
                           truncation: bool = True) -> list[EmbeddingUnit]:
        ...

    def encode_texts(self, 
                           texts: list[str], 
                           include_metadata: bool = False,
                           input_type: Optional[Literal["query", "document"]] = "document",
                           truncation: bool = True,) -> Union[list[EmbeddingUnit], EmbeddingResponse]:
        """
        Encode texts into embeddings.

        Args:
            texts (list[str]): The texts to encode.
            input_type (Optional[Literal["query", "document"]]): The type of input. Can be "query" or "document". Default is "document".
            truncation (bool): Whether to truncate the texts. Default is True.
            include_metadata (bool): Whether to include metadata in the response.

        Returns:
            Returns:
            If `include_metadata` is `True`, return an `EmbeddingResponse` object containing the embedding with metadata.
            
            If `include_metadata` is `False`, return an `EmbeddingUnit` object containing the embedding.
        """
        response = self._client.embed(
            texts=texts,
            model=self.embedding_model,
            input_type=input_type,
            truncation=truncation,
        )

        embedding_units: list[EmbeddingUnit] = []

        for idx, embedding in enumerate(response.embeddings):
            if self.encoding_format == "base64":
                embedding = self._handle_base64_encoding(embedding)
            
            embedding_units.append(
                EmbeddingUnit(
                    index=idx,
                    object="embedding",
                    content=texts[idx],
                    embedding=embedding,
                    type=self._encoding_format
                )
            )

        if include_metadata:
            return EmbeddingResponse(
                embeddings=embedding_units,
                embedding_model=self.embedding_model,
                total_tokens=response.total_tokens
            )
        
        return embedding_units

    def tokenize_texts(self, texts: list[str]) -> list[list[str]]:
        """
        Tokenize a list of texts into a list of tokens.
        Args:
            texts: A list of texts to tokenize.
        Returns:
            A list of tokens lists for each text.
        """
        tokenized = self._client.tokenize(texts, model=self.embedding_model)

        tokenized_texts = [item for item in tokenized if isinstance(item, list)] # type: ignore
        
        return tokenized_texts # type: ignore
