from openagentkit.core.models.responses.audio_response import AudioResponse as AudioResponse
from openagentkit.core.models.responses.tool_response import ToolCall as ToolCall
from openagentkit.core.models.responses.usage_responses import UsageResponse as UsageResponse
from pydantic import BaseModel
from typing import Any

class OpenAgentResponse(BaseModel):
    role: str
    reasoning: str | None
    content: str | BaseModel | dict[str, str] | None
    tool_calls: list[ToolCall] | None
    tool_results: list[Any] | None
    refusal: str | None
    audio: AudioResponse | None
    usage: UsageResponse | None
