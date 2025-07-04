import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from typing import AsyncGenerator

class AsyncBaseSTTModel(ABC, metaclass=abc.ABCMeta):
    model: Incomplete
    def __init__(self, model: str) -> None: ...
    @abstractmethod
    async def speech_to_text(self, audio_bytes: bytes) -> str: ...
    @abstractmethod
    async def stream_speech_to_text(self, audio_bytes: bytes) -> AsyncGenerator[str, None]: ...

class AsyncBaseTTSModel(ABC, metaclass=abc.ABCMeta):
    model: Incomplete
    def __init__(self, model: str) -> None: ...
    @abstractmethod
    async def text_to_speech(self, text: str) -> bytes: ...
    @abstractmethod
    async def stream_text_to_speech(self, text: str) -> AsyncGenerator[bytes, None]: ...
