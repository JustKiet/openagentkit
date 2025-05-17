import io
from typing import Any

class NamedBytesIO(io.BytesIO):
    def __init__(self, *args: Any, name="audio.wav", **kwargs: Any): # type: ignore
        super().__init__(*args, **kwargs) # type: ignore
        self.name = name
