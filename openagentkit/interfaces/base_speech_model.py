from abc import ABC, abstractmethod

class BaseSpeechModel(ABC):
    """
    An abstract base class for speech models.
    
    ## Methods:
        `text_to_speech()`: An abstract method to convert text to speech.
    """
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