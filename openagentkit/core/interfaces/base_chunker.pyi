import abc
from abc import ABC, abstractmethod

class BaseChunker(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def get_chunks(self, text: str) -> list[str]: ...
