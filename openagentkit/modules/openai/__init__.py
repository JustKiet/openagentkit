"""
A module for OpenAI related services and modules.

## Modules:
    - `openai_llm_service`: A module for OpenAI LLM service.
    - `openai_speech_service`: A module for OpenAI speech service.
    - `openai_executor`: A module for OpenAI executor.
    - `async_openai_executor`: A module for OpenAI asynchronous executor.
    - `async_openai_llm_service`: A module for OpenAI asynchronous LLM service.
    - `openai_embedding_service`: A module for OpenAI embedding service.
"""
from typing import TypeAlias, Literal

OpenAIEmbeddingModels: TypeAlias = Literal[
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]

OpenAIEmbeddingEncodings: TypeAlias = Literal[
    "cl100k_base",
]

OpenAIEncodingFormats: TypeAlias = Literal[
    "float",
    "base64",
]

OpenAIAudioFormats: TypeAlias = Literal[
    'wav', 'mp3', 'flac', 'opus', 'pcm16'
]

OpenAIAudioVoices: TypeAlias = Literal[
    "alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"
]

from .openai_llm_service import OpenAILLMService # type: ignore
from .openai_speech_service import OpenAISTTService # type: ignore
from .openai_agent import OpenAIAgent # type: ignore
from .async_openai_agent import AsyncOpenAIAgent # type: ignore
from .async_openai_llm_service import AsyncOpenAILLMService # type: ignore
from .openai_embedding_service import OpenAIEmbeddingModel # type: ignore