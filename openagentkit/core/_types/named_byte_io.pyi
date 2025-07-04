import io
from _typeshed import Incomplete
from typing import Any

class NamedBytesIO(io.BytesIO):
    name: Incomplete
    def __init__(self, *args: Any, name: str = 'audio.wav', **kwargs: Any) -> None: ...
