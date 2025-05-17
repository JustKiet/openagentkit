from openagentkit.core.interfaces.base_speech_model import BaseSpeechModel
import requests
import os
import io
import wave
from typing import Optional, Any

class LightningSpeechService(BaseSpeechModel):
    def __init__(self,
                 voice_id: str = "arman",
                 add_wav_header: bool = True,
                 sample_rate: int = 16000,
                 speed: float = 0.95,
                 api_key: Optional[str] = os.getenv("SMALLEST_API_KEY"),):
        self.voice_id = voice_id
        self.add_wav_header = add_wav_header
        self.sample_rate = sample_rate
        self.speed = speed
        self.api_key = api_key if api_key else os.getenv("SMALLEST_API_KEY")
    
    def _chunk_text(self, text: str, max_chunk_size: int = 200) -> list[str]:
        """
        Chunks text with a maximum size of 200 characters, preferring to break at punctuation marks.
        
        Args:
            text (str): Input text to be chunked
            max_chunk_size (int): Maximum size of each chunk (default: 200)
            
        Returns:
            list: List of text chunks
        """
        chunks: list[str] = []
        while text:
            if len(text) <= max_chunk_size:
                chunks.append(text)
                break
                
            # Look for punctuation within the last 50 characters of the max chunk size
            chunk_end = max_chunk_size
            punctuation_marks = '.,:;।!?'
            
            # Search backward from max_chunk_size for punctuation
            found_punct = False
            for i in range(chunk_end, max(chunk_end - 50, 0), -1):
                if i < len(text) and text[i] in punctuation_marks:
                    chunk_end = i + 1  # Include the punctuation mark
                    found_punct = True
                    break
            
            # If no punctuation found, look for space
            if not found_punct:
                for i in range(chunk_end, max(chunk_end - 50, 0), -1):
                    if i < len(text) and text[i].isspace():
                        chunk_end = i
                        break
                # If no space found, force break at max_chunk_size
                if not found_punct and chunk_end == max_chunk_size:
                    chunk_end = max_chunk_size
            
            # Add chunk and remove it from original text
            chunks.append(text[:chunk_end].strip())
            text = text[chunk_end:].strip()
        
        return chunks
    
    def text_to_speech(self, text: str) -> bytes:
        url = "https://waves-api.smallest.ai/api/v1/lightning/get_speech"
        
        chunks = self._chunk_text(text)
        raw_audio_data: list[bytes] = []  # List to store PCM data
        sample_width: Optional[int] = None
        channels: Optional[int] = None
        frame_rate: Optional[int] = None
        
        for i, chunk in enumerate(chunks):
            payload: dict[str, Any] = {
                "text": chunk,
                "voice_id": self.voice_id,
                "add_wav_header": self.add_wav_header,
                "sample_rate": self.sample_rate,
                "speed": self.speed,
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                if response.headers.get("Content-Type") == "audio/wav":
                    wav_io = io.BytesIO(response.content)
                    with wave.open(wav_io, "rb") as wav:
                        if i == 0:
                            sample_width = wav.getsampwidth()
                            channels = wav.getnchannels()
                            frame_rate = wav.getframerate()
                        raw_audio_data.append(
                            wav.readframes(wav.getnframes())
                        )
            else:
                raise Exception(f"Failed to generate Lightning speech: {response.text}")
        
        if channels is None or sample_width is None or frame_rate is None:
            raise ValueError("Failed to retrieve audio properties from the response.")
        
        output_wav = io.BytesIO()
        with wave.open(output_wav, "wb") as out_wav:
            out_wav.setnchannels(channels)
            out_wav.setsampwidth(sample_width)
            out_wav.setframerate(frame_rate)
            out_wav.writeframes(b"".join(raw_audio_data))
            
        return output_wav.getvalue()