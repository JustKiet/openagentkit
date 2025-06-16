from typing import AsyncGenerator
from abc import ABC, abstractmethod

class AsyncBaseSTTModel(ABC):
    def __init__(self,
                 model: str) -> None:
        """
        Initializes the AsyncBaseSTTModel with a model name.
        Args:
            model (str): The name of the speech model to use.
        """
        self.model = model

    @abstractmethod
    async def speech_to_text(self, audio_bytes: bytes) -> str:
        """
        An abstract method to convert speech audio bytes to text.

        Args:
            audio_bytes (bytes): The audio bytes to convert to text.

        Returns:
            str: The text transcription of the audio data.
        """
        raise NotImplementedError
    
    @abstractmethod
    async def stream_speech_to_text(self, audio_bytes: bytes) -> AsyncGenerator[str, None]:
        """
        An abstract method to stream speech audio bytes to text.

        Args:
            audio_bytes (bytes): The audio bytes to convert to text.

        Returns:
            AsyncGenerator[str, None]: A generator that yields text transcriptions of the audio data.
        """
        raise NotImplementedError
    
class AsyncBaseTTSModel(ABC):
    def __init__(self,
                 model: str) -> None:
        """
        Initializes the BaseTTSModel with a model name.
        
        Args:
            model (str): The name of the text-to-speech model to use.
        """
        self.model = model

    @abstractmethod
    async def text_to_speech(
        self, 
        text: str,
    ) -> bytes:
        """
        Convert text to speech audio bytes.

        Args:
            text (str): The text to convert to speech.

        Returns:
            bytes: The audio bytes of the spoken text.
        """
        raise NotImplementedError
    
    @abstractmethod
    async def stream_text_to_speech(
        self, 
        text: str,
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream text to speech audio bytes.

        Args:
            text (str): The text to convert to speech.

        Returns:
            AsyncGenerator[bytes, None]: A generator that yields audio bytes of the spoken text.
        """
        raise NotImplementedError