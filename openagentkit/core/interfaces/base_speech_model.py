from abc import ABC, abstractmethod

class BaseSpeechModel(ABC):
    """
    An abstract base class for speech models.
    
    ## Methods:
        `text_to_speech()`: An abstract method to convert text to speech.
    """
    @abstractmethod
    def speech_to_text(self, audio_bytes: bytes) -> str:
        """
        An abstract method to convert speech audio bytes to text.

        Args:
            audio_bytes (bytes): The audio bytes to convert to text.

        Returns:
            str: The text transcription of the audio data.
        """
        raise NotImplementedError

    @abstractmethod
    def text_to_speech(self, text: str) -> bytes:
        """
        An abstract method to convert text to speech.

        Args:
            text (str): The text to convert to speech.

        Returns:
            bytes: The audio data in bytes.
        """
        raise NotImplementedError