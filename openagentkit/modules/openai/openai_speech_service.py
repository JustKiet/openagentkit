from typing import Optional, Literal, AsyncGenerator, Union
from openagentkit.interfaces import BaseSpeechModel
from openagentkit._types import NamedBytesIO
from openai import AsyncOpenAI, OpenAI
import wave
from loguru import logger
import tempfile
import os
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
            
            # For WebM, MP3, OGG, etc., save to a temp file with appropriate extension
            if audio_format in ["webm", "mp3", "ogg", "m4a", "mpeg", "mpga", "flac"]:
                # Try FFmpeg conversion first if it's a WebM from browser (often more reliable)
                if audio_format == "webm":
                    logger.info("Attempting to convert WebM to WAV using FFmpeg")
                    converted_wav = AudioUtility.convert_audio_format(audio_data, "webm", "wav")
                    if converted_wav:
                        logger.info("WebM conversion successful, using converted WAV")
                        response = self._client.audio.transcriptions.create(
                            model=self.stt_model,
                            file=NamedBytesIO(converted_wav, name="converted_audio.wav"),
                        )
                        transcription = response.text
                        logger.info(f"Transcription: {transcription}")
                        return transcription
                
                # Create temp file with appropriate extension
                with tempfile.NamedTemporaryFile(suffix=f'.{audio_format}', delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_path = temp_file.name
                
                logger.info(f"Saved {audio_format} audio to temporary file: {temp_path}")
                
                # Create a file object for OpenAI API
                try:
                    with open(temp_path, 'rb') as f:
                        response = self._client.audio.transcriptions.create(
                            model=self.stt_model,
                            file=f,
                        )
                    
                    # Clean up
                    os.unlink(temp_path)
                    
                    # Send transcription as an event
                    transcription = response.text
                    logger.info(f"Transcription: {transcription}")
                    return transcription
                except Exception as inner_e:
                    logger.error(f"Error processing {audio_format} file: {inner_e}")
                    
                    # Fallback: Try converting to WAV first
                    logger.info(f"Attempting fallback: Converting {audio_format} to WAV")
                    try:
                        os.unlink(temp_path)  # Clean up the original file
                        
                        # Convert to WAV using our utility
                        converted_wav = AudioUtility.convert_audio_format(audio_data, audio_format, "wav")
                        if converted_wav:
                            response = self._client.audio.transcriptions.create(
                                model=self.stt_model,
                                file=NamedBytesIO(converted_wav, name="converted_audio.wav"),
                            )
                            transcription = response.text
                            logger.info(f"Fallback transcription successful: {transcription}")
                            return transcription
                        else:
                            return "Sorry, I couldn't process the audio after conversion attempts."
                    except Exception as fallback_e:
                        logger.error(f"Fallback conversion failed: {fallback_e}")
                        return "Sorry, I couldn't process the audio format."
            
            # For wav format
            elif audio_format == "wav":
                # Check if it's a valid WAV
                if AudioUtility.validate_wav(audio_data):
                    response = self._client.audio.transcriptions.create(
                        model=self.stt_model,
                        file=NamedBytesIO(audio_data, name="audio.wav"),
                    )
                    transcription = response.text
                    logger.info(f"Transcription: {transcription}")
                    return transcription
                else:
                    logger.warning("Invalid WAV file received")
                    return "Sorry, I couldn't process the invalid WAV audio."
            
            # For raw PCM or unknown formats, convert to WAV
            else:
                logger.info("Converting to WAV from raw or unknown format")
                wav_data = AudioUtility.raw_bytes_to_wav(audio_data).getvalue()
                
                try:
                    response = self._client.audio.transcriptions.create(
                        model=self.stt_model,
                        file=NamedBytesIO(wav_data, name="audio.wav"),
                    )
                    transcription = response.text
                    logger.info(f"Transcription: {transcription}")
                    return transcription
                except Exception as wav_e:
                    logger.error(f"Error processing converted WAV: {wav_e}")
                    
                    # Last resort: Just try to dump the raw bytes as MP3 (sometimes works)
                    logger.info("Last resort attempt: saving as MP3")
                    try:
                        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                            temp_file.write(audio_data)
                            temp_path = temp_file.name
                        
                        with open(temp_path, 'rb') as f:
                            response = self._client.audio.transcriptions.create(
                                model=self.stt_model,
                                file=f,
                            )
                        
                        os.unlink(temp_path)
                        
                        transcription = response.text
                        logger.info(f"Last resort transcription successful: {transcription}")
                        return transcription
                    except Exception as last_e:
                        logger.error(f"Last resort failed: {last_e}")
                        return "Sorry, I couldn't process the audio after multiple attempts."
        
        except Exception as e:
            logger.error(f"Error in speech_to_text: {e}")
            print(e)
            return "Sorry, I couldn't transcribe the audio."
    
    def text_to_speech(self, 
                       message: str,
                       response_format: Optional[str] = "wav",
                       ) -> bytes:
        """
        Convert text to speech.

        Args:
            message (str): The text to convert to speech.
            response_format (Optional[str]): The format to use in the response.

        Returns:
            bytes: The audio data in bytes.
        """
        response = self._client.audio.speech.create(
            model="tts-1",
            voice=self.voice,
            input=message,
            response_format=response_format,
        )
        return response.content
    