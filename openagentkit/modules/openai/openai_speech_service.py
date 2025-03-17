from typing import Optional, Literal
from openagentkit.interfaces import BaseSpeechModel
from openai import OpenAI

class OpenAISpeechService(BaseSpeechModel):
    def __init__(self,
                 client: OpenAI,
                 voice: Optional[Literal["alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"]] = "nova",
                 *args,
                 **kwargs,):
        self._client = client
        self.voice = voice
        
    def text_to_speech(self, 
                       message: str,
                       response_format: Optional[str] = "wav") -> bytes:
        """Convert text to speech."""
        response = self._client.audio.speech.create(
            model="tts-1",
            voice=self.voice,
            input=message,
            response_format=response_format,
        )
        return response.content
    