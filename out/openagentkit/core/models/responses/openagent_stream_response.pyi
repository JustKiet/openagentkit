from openagentkit.core.models.responses.tool_response import ToolCall as ToolCall, ToolCallResult as ToolCallResult
from openagentkit.core.models.responses.usage_responses import UsageResponse as UsageResponse
from pydantic import BaseModel
from typing import Literal

class OpenAgentStreamingResponse(BaseModel):
    role: str
    index: int | None
    delta_reasoning: str | None
    delta_content: str | None
    delta_audio: str | None
    tool_calls: list[ToolCall] | None
    tool_results: list[ToolCallResult] | None
    tool_notification: str | None
    reasoning: str | None
    content: str | None
    finish_reason: Literal['stop', 'length', 'tool_calls', 'tool_results', 'content_filter', 'function_call', 'transcription'] | None
    usage: UsageResponse | None
