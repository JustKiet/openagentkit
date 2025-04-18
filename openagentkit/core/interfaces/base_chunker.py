from abc import ABC, abstractmethod

class BaseChunker(ABC):
    @abstractmethod
    def get_chunks(self, text: str) -> list:
        """
        Splits the input text into chunks.

        Args:
            text (str): The input text to be chunked.

        Returns:
            list: A list of chunks.
        """
        raise NotImplementedError