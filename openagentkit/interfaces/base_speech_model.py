from abc import ABC, abstractmethod

class BaseSpeechModel(ABC):
    @abstractmethod
    def text_to_speech(self) -> bytes:
        raise NotImplementedError