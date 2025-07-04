from openagentkit.core.models.responses.tool_response import *
from _typeshed import Incomplete
from openagentkit.core.interfaces import BaseLLMModel as BaseLLMModel
from openagentkit.core.models.responses import CompletionTokensDetails as CompletionTokensDetails, OpenAgentResponse as OpenAgentResponse, OpenAgentStreamingResponse as OpenAgentStreamingResponse, PromptTokensDetails as PromptTokensDetails, UsageResponse as UsageResponse
from openagentkit.core.models.responses.audio_response import AudioResponse as AudioResponse
from openagentkit.core.tools.base_tool import Tool as Tool
from openagentkit.core.tools.tool_handler import ToolHandler as ToolHandler
from openagentkit.modules.openai import OpenAIAudioFormats as OpenAIAudioFormats, OpenAIAudioVoices as OpenAIAudioVoices
from openai import OpenAI
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall as ChoiceDeltaToolCall
from pydantic import BaseModel
from typing import Any, Generator

logger: Incomplete

class OpenAILLMService(BaseLLMModel):
    def __init__(self, client: OpenAI | None = None, model: str = 'gpt-4o-mini', tools: list[Tool] | None = None, api_key: str | None = ..., temperature: float | None = 0.3, max_tokens: int | None = None, top_p: float | None = None) -> None: ...
    @property
    def client(self) -> OpenAI | None: ...
    @property
    def api_key(self) -> str | None: ...
    @property
    def tool_handler(self) -> ToolHandler: ...
    @tool_handler.setter
    def tool_handler(self, value: ToolHandler) -> None: ...
    @property
    def tools(self): ...
    def clone(self) -> OpenAILLMService: ...
    def model_generate(self, messages: list[dict[str, str]], response_schema: type[BaseModel] | None = None, temperature: float | None = None, max_tokens: int | None = None, top_p: float | None = None, tools: list[dict[str, Any]] | None = None, audio: bool | None = False, audio_format: OpenAIAudioFormats | None = 'pcm16', audio_voice: OpenAIAudioVoices | None = None, reasoning_effort: Literal['low', 'medium', 'high'] | None = None, **kwargs: Any) -> OpenAgentResponse: ...
    def model_stream(self, messages: list[dict[str, str]], response_schema: type[BaseModel] | None = None, temperature: float | None = None, max_tokens: int | None = None, top_p: float | None = None, tools: list[dict[str, Any]] | None = None, audio: bool | None = False, audio_format: OpenAIAudioFormats | None = 'pcm16', audio_voice: OpenAIAudioVoices | None = 'alloy', reasoning_effort: Literal['low', 'medium', 'high'] | None = None, **kwargs: Any) -> Generator[OpenAgentStreamingResponse, None, None]: ...
