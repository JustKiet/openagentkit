import io
from _typeshed import Incomplete
from typing import TypeAlias

logger: Incomplete
AudioFormat: TypeAlias

class AudioUtility:
    @staticmethod
    def detect_audio_format(audio_bytes: bytes) -> AudioFormat: ...
    @staticmethod
    def validate_wav(wav_bytes: bytes) -> bool: ...
    @staticmethod
    def raw_bytes_to_wav(raw_audio_bytes: bytes, sample_rate: int = 16000, num_channels: int = 1, sample_width: int = 2) -> io.BytesIO: ...
    @staticmethod
    def convert_audio_format(audio_bytes: bytes, source_format: str, target_format: str = 'wav') -> bytes | None: ...
