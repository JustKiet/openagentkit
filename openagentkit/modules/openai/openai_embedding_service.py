import tiktoken
from openai import OpenAI
from openai.types import CreateEmbeddingResponse

class OpenAIEmbeddingModel:
    def __init__(self, 
                 embedding_model: str = "text-embedding-3-small",
                 embedding_encoding: str = "cl100k_base",
                 max_token: int = 8000):
        self.client = OpenAI()
        self.embedding_model = embedding_model
        self.embedding_encoding = embedding_encoding
        self.max_token = max_token

    def encode(self, texts: list[str]) -> CreateEmbeddingResponse:
        formatted_texts = []

        for text in texts:
            text = text.replace("\n", " ")
            formatted_texts.append(text)

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=formatted_texts,
            encoding_format=self.embedding_encoding,
        )
        
        return response
    
if __name__ == "__main__":
    texts = ["Hello, world!", "This is a test."]
    embedding_model = OpenAIEmbeddingModel()
    response = embedding_model.encode(texts)
    print(response)
