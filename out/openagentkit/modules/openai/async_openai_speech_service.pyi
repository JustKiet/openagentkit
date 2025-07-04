import abc
from _typeshed import Incomplete
from openagentkit.core._types import NamedBytesIO as NamedBytesIO
from openagentkit.core.interfaces import BaseSTTModel as BaseSTTModel, BaseTTSModel as BaseTTSModel
from openagentkit.core.utils.audio_utils import AudioUtility as AudioUtility
from openagentkit.modules.openai import OpenAIAudioVoices as OpenAIAudioVoices
from openai import OpenAI as OpenAI
from typing import Generator, Literal

logger: Incomplete

class AsyncOpenAISTTService(BaseSTTModel, metaclass=abc.ABCMeta):
    voice: Incomplete
    model: Incomplete
    def __init__(self, client: OpenAI, voice: OpenAIAudioVoices = 'nova', model: Literal['whisper-1', 'gpt-4o-transcribe', 'gpt-4o-mini-transcribe'] = 'whisper-1') -> None: ...
    def speech_to_text(self, audio_bytes: bytes) -> str: ...

class OpenAITTSService(BaseTTSModel):
    voice: Incomplete
    model: Incomplete
    def __init__(self, client: OpenAI, voice: OpenAIAudioVoices = 'nova', model: Literal['tts-1'] = 'tts-1') -> None: ...
    def text_to_speech(self, text: str, response_format: Literal['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm'] = 'wav') -> bytes: ...
    def stream_text_to_speech(self, text: str, chunk_size: int = 1024, response_format: Literal['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm'] = 'pcm', speed: float = 1.0) -> Generator[bytes, None, None]: ...
