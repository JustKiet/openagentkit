from typing import Optional, Literal
from openagentkit.interfaces import BaseSpeechModel
from openagentkit._types import NamedBytesIO
from openai import OpenAI
import wave
from loguru import logger
import io

from openagentkit.utils.audio_utils import AudioUtility

class OpenAISpeechService(BaseSpeechModel):
    def __init__(self,
                 client: OpenAI,
                 voice: Optional[Literal["alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"]] = "nova",
                 stt_model: Optional[str] = "whisper-1",
                 *args,
                 **kwargs,):
        self._client = client
        self.voice = voice
        self.stt_model = stt_model
    
    def speech_to_text(self, audio_data: bytes) -> str:
        """Convert speech audio data to text using OpenAI's API."""
        try:
            # Check if it's a WebM file (common from browser recording)
            is_webm = AudioUtility.detect_audio_format(audio_data) == "webm"
            
            if is_webm:
                logger.info("Detected WebM format audio")
                # For WebM, we need to save it to a temporary file and use it directly
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_path = temp_file.name
                
                # Create a named file for OpenAI API
                with open(temp_path, 'rb') as f:
                    response = self._client.audio.transcriptions.create(
                        model=self.stt_model,
                        file=f,  # OpenAI can handle WebM directly
                    )
                
                # Clean up
                import os
                os.unlink(temp_path)
                
                return response.text
            
            # If not WebM, try the normal WAV conversion
            elif not AudioUtility.validate_wav(audio_data):
                audio_data = AudioUtility.raw_bytes_to_wav(audio_data).getvalue()
                
                if AudioUtility.validate_wav(audio_data):
                    response = self._client.audio.transcriptions.create(
                        model=self.stt_model,
                        file=NamedBytesIO(audio_data, name="audio.wav"),
                    )
                    return response.text
                else:
                    return "Sorry, I couldn't process the audio."
            else:
                # It's already a valid WAV
                response = self._client.audio.transcriptions.create(
                    model=self.stt_model,
                    file=NamedBytesIO(audio_data, name="audio.wav"),
                )
                return response.text
        except Exception as e:
            logger.error(f"Error in speech_to_text: {e}")
            return "Sorry, I couldn't transcribe the audio."
    
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
    