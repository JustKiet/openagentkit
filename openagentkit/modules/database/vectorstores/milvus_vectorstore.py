from pymilvus import MilvusClient
from openagentkit.modules.openai import OpenAIEmbeddingModel
from openagentkit.core.interfaces import BaseEmbeddingModel
import numpy as np
import uuid

class MilvusVectorStore:
    def __init__(self,
                 embedding_model: BaseEmbeddingModel,
                 uri: str = "milvus://localhost:19530",
                 collection_name: str = "demo_collection",
                 password: str = "",
                 username: str = "",
                 token: str = "",
                 timeout: int = None,):
        self.client = MilvusClient(
            uri=uri,
            collection_name=collection_name,
            password=password,
            username=username,
            token=token,
            timeout=timeout,
        )
        self.embedding_model = embedding_model

    def create_collection(self, collection_name: str, dimension: int) -> str:
        if self.client.has_collection(collection_name=collection_name):
            raise ValueError(f"Collection {collection_name} already exists.")
        self.client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
        )
        return f"Collection {collection_name} created successfully."
    
    def add_documents(self, documents: list[str], collection_name: str) -> str:
        if not self.client.has_collection(collection_name=collection_name):
            raise ValueError(f"Collection {collection_name} does not exist.")
        embeddings = self.embedding_model.encode_texts(documents)

        data = [
            {
                "id": np.int64(uuid.uuid4().int & (2**63 - 1)),
                "vector": embedding.embedding,
                "content": embedding.content,
            }
            for embedding in embeddings
        ]

        self.client.insert(
            collection_name=collection_name,
            data=data,
        )
        return f"{len(embeddings)} documents added to {collection_name}."

