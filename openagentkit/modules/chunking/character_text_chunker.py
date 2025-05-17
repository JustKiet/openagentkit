from openagentkit.core.interfaces import BaseChunker

class CharacterTextChunker(BaseChunker):
    """
    A simple text chunker that splits text into chunks based on a specified character limit.
    """

    def __init__(self, 
                 chunk_size: int = 100,
                 chunk_overlap: int = 0):
        """
        Initializes the CharacterTextChunker with a specified chunk size.

        Args:
            chunk_size (int): The maximum size of each chunk in characters.
        """
        if chunk_size <= 0:
            raise ValueError("Chunk size must be a positive integer.")
        if chunk_overlap < 0:
            raise ValueError("Chunk overlap must be a non-negative integer.")
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size.")
        
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @property
    def chunk_size(self) -> int:
        """
        Returns the chunk size.

        Returns:
            int: The chunk size.
        """
        return self._chunk_size
    
    @property
    def chunk_overlap(self) -> int:
        """
        Returns the chunk overlap.

        Returns:
            int: The chunk overlap.
        """
        return self._chunk_overlap

    def get_chunks(self, text: str) -> list[str]:
        """
        Splits the input text into chunks based on the specified character limit.

        Args:
            text (str): The input text to be chunked.

        Returns:
            list: A list of chunks.
        """
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks
    