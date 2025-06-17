from typing import Optional, Literal, Generator
from openagentkit.core.interfaces import BaseTTSModel, BaseSTTModel
from openagentkit.core._types import NamedBytesIO
from openai import OpenAI
import logging

from openagentkit.core.utils.audio_utils import AudioUtility
from openagentkit.modules.openai import OpenAIAudioVoices

logger = logging.getLogger(__name__)

class AsyncOpenAISTTService(BaseSTTModel):
    def __init__(
        self,
        client: OpenAI,
        voice: OpenAIAudioVoices = "nova",
        model: Literal["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"] = "whisper-1",
    ) -> None:
        self._client = client
        self.voice = voice
        self.model = model
    
    def _transcribe_audio(self, file_obj: bytes, file_name: Optional[str] = None):
        """Helper method to call OpenAI transcription API with consistent parameters"""
        if file_name and isinstance(file_obj, bytes):
            file_obj = NamedBytesIO(file_obj, name=file_name) # type: ignore
            
        response = self._client.audio.transcriptions.create(
            model=self.model,
            file=file_obj,
        )
        return response.text
    
    def speech_to_text(
        self, 
        audio_bytes: bytes
    ) -> str:
        """
        Convert speech audio bytes to text using OpenAI's API.

        Args:
            audio_bytes (bytes): The audio bytes to convert to text.

        Returns:
            str: The text transcription of the audio data.
        """
        try:
            # Detect the audio format
            audio_format = AudioUtility.detect_audio_format(audio_bytes)
            logger.info(f"Detected audio format: {audio_format}")
            
            # Direct handling for WAV format
            if audio_format == "wav" and AudioUtility.validate_wav(audio_bytes):
                return self._transcribe_audio(audio_bytes, "audio.wav")
                
            # WebM conversion (most common from browsers)
            if audio_format == "webm":
                converted_wav = AudioUtility.convert_audio_format(audio_bytes, "webm", "wav")
                if converted_wav:
                    return self._transcribe_audio(converted_wav, "converted_audio.wav")
            
            # Handle common audio formats - first try direct approach
            if audio_format in ["mp3", "ogg", "m4a", "mpeg", "mpga", "flac"]:
                raise ValueError(f"Unsupported audio format: {audio_format}. Please convert to WAV, PCM, or WebM.")
            
            # Raw PCM or unknown formats - convert to WAV
            wav_data = AudioUtility.raw_bytes_to_wav(audio_bytes).getvalue()
            return self._transcribe_audio(wav_data, "audio.wav")
            
        except Exception as e:
            logger.error(f"Error in speech_to_text: {e}")
            return "Sorry, I couldn't transcribe the audio."
    
class OpenAITTSService(BaseTTSModel):
    def __init__(
        self,
        client: OpenAI,
        voice: OpenAIAudioVoices = "nova",
        model: Literal["tts-1"] = "tts-1",
    ) -> None:
        self._client = client
        self.voice = voice
        self.model = model
    
    def text_to_speech(
        self, 
        text: str,
        response_format: Literal['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm'] = "wav",
    ) -> bytes:
        """
        Convert text to speech using OpenAI's API.

        Args:
            text (str): The text to convert to speech.
            response_format (Literal['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm']): The format to use in the response.

        Returns:
            bytes: The audio data in bytes.
        """

        response = self._client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format=response_format,
        )
        return response.content
    
    def stream_text_to_speech(
        self,
        text: str,
        chunk_size: int = 1024,
        response_format: Literal['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm'] = "pcm",
        speed: float = 1.0,
    ) -> Generator[bytes, None, None]:
        """
        Stream text to speech audio bytes using OpenAI's API.

        Args:
            text (str): The text to convert to speech.
            chunk_size (int): The size of each audio chunk to yield.
            response_format (Literal['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm']): The format to use in the response.
            speed (float): The speed of the speech. 1.0 is normal speed, 0.5 is half speed, etc.

        Returns:
            Generator[bytes, None, None]: A generator that yields audio chunks.
        """
        with self._client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format=response_format,
            speed=speed,
        ) as stream:
            for audio_chunk in stream.iter_bytes(chunk_size=chunk_size):
                yield audio_chunk
