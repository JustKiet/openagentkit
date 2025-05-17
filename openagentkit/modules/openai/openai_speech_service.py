from typing import Optional, Literal
from openagentkit.core.interfaces import BaseSpeechModel
from openagentkit.core._types import NamedBytesIO
from openai import OpenAI
from loguru import logger
import tempfile
import os

from openagentkit.core.utils.audio_utils import AudioUtility
from openagentkit.modules.openai import OpenAIAudioVoices

class OpenAISpeechService(BaseSpeechModel):
    def __init__(self,
                 client: OpenAI,
                 voice: OpenAIAudioVoices = "nova",
                 stt_model: Literal["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"] = "whisper-1",
                 ) -> None:
        self._client = client
        self.voice = voice
        self.stt_model = stt_model
    
    def _transcribe_audio(self, file_obj: bytes, file_name: Optional[str] = None):
        """Helper method to call OpenAI transcription API with consistent parameters"""
        if file_name and isinstance(file_obj, bytes):
            file_obj = NamedBytesIO(file_obj, name=file_name) # type: ignore
            
        response = self._client.audio.transcriptions.create(
            model=self.stt_model,
            file=file_obj,
        )
        return response.text
    
    def speech_to_text(self, audio_data: bytes) -> str:
        """
        Convert speech audio data to text using OpenAI's API.

        Args:
            audio_data (bytes): The audio data to convert to text.

        Returns:
            str: The text transcription of the audio data.
        """
        try:
            # Detect the audio format
            audio_format = AudioUtility.detect_audio_format(audio_data)
            logger.info(f"Detected audio format: {audio_format}")
            
            # Direct handling for WAV format
            if audio_format == "wav" and AudioUtility.validate_wav(audio_data):
                return self._transcribe_audio(audio_data, "audio.wav")
                
            # WebM conversion (most common from browsers)
            if audio_format == "webm":
                converted_wav = AudioUtility.convert_audio_format(audio_data, "webm", "wav")
                if converted_wav:
                    return self._transcribe_audio(converted_wav, "converted_audio.wav")
            
            # Handle common audio formats - first try direct approach
            if audio_format in ["mp3", "ogg", "m4a", "mpeg", "mpga", "flac", "webm"]:
                temp_path = None
                try:
                    # Create temp file with appropriate extension
                    with tempfile.NamedTemporaryFile(suffix=f'.{audio_format}', delete=False) as temp_file:
                        temp_file.write(audio_data)
                        temp_path = temp_file.name
                    
                    # Try direct transcription
                    with open(temp_path, 'rb') as f:
                        transcription = self._transcribe_audio(f) # type: ignore
                        
                    return transcription
                    
                except Exception:
                    # Try converting to WAV as fallback
                    if temp_path:
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                            
                    # Try WAV conversion
                    converted_wav = AudioUtility.convert_audio_format(audio_data, audio_format, "wav")
                    if converted_wav:
                        return self._transcribe_audio(converted_wav, "converted_audio.wav")
            
            # Raw PCM or unknown formats - convert to WAV
            wav_data = AudioUtility.raw_bytes_to_wav(audio_data).getvalue()
            try:
                return self._transcribe_audio(wav_data, "audio.wav")
            except Exception:
                # Last resort for any format - try as MP3
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                        temp_file.write(audio_data)
                        temp_path = temp_file.name
                    
                    with open(temp_path, 'rb') as f:
                        return self._transcribe_audio(f) # type: ignore
                finally:
                    if temp_path:
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
            
            return "Sorry, I couldn't process the audio after multiple attempts."
            
        except Exception as e:
            logger.error(f"Error in speech_to_text: {e}")
            return "Sorry, I couldn't transcribe the audio."
    
    def text_to_speech(self, 
                       text: str,
                       response_format: Literal['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm'] = "wav",
                       ) -> bytes:
        """
        Convert text to speech.

        Args:
            text (str): The text to convert to speech.
            response_format (Literal['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm']): The format to use in the response.

        Returns:
            bytes: The audio data in bytes.
        """

        response = self._client.audio.speech.create(
            model="tts-1",
            voice=self.voice,
            input=text,
            response_format=response_format,
        )
        return response.content
    