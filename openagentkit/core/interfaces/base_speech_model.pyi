import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from typing import Generator

class BaseSTTModel(ABC, metaclass=abc.ABCMeta):
    model: Incomplete
    def __init__(self, model: str) -> None: ...
    @abstractmethod
    def speech_to_text(self, audio_bytes: bytes) -> str: ...
    @abstractmethod
    def stream_speech_to_text(self, audio_bytes: bytes) -> Generator[str, None, None]: ...

class BaseTTSModel(ABC, metaclass=abc.ABCMeta):
    model: Incomplete
    def __init__(self, model: str) -> None: ...
    @abstractmethod
    def text_to_speech(self, text: str) -> bytes: ...
    @abstractmethod
    def stream_text_to_speech(self, text: str) -> Generator[bytes, None, None]: ...
