from pydantic import BaseModel
from typing import Literal, Optional, List
from openagentkit.core.models.responses.usage_responses import UsageResponse
from openagentkit.core.models.responses.tool_response import ToolCall, ToolCallResult

class OpenAgentStreamingResponse(BaseModel):
    """
    The response schema for OpenAgentKit streaming responses.

    Schema:
        ```python
        class OpenAgentStreamingResponse(BaseModel):
            role: str
            index: Optional[int] = None
            delta_reasoning: Optional[str] = None
            delta_content: Optional[str] = None
            delta_audio: Optional[str] = None
            tool_calls: Optional[List[ToolCallResult]] = None
            tool_notification: Optional[str] = None
            content: Optional[str] = None
            reasoning: Optional[str] = None
            finish_reason: Optional[Literal["stop", "length", "tool_calls", "tool_results", "content_filter", "function_call", "transcription"]] = None
            usage: Optional[UsageResponse] = None
        ```
    Where:
        - `role`: The role of the response.
        - `index`: The index of the response.
        - `delta_reasoning`: The delta reasoning of the response.
        - `delta_content`: The delta content of the response.
        - `delta_audio`: The delta audio in base64 format of the response.
        - `tool_calls`: The tool calls of the response.
        - `tool_notification`: The tool notification of the response.
        - `reasoning`: The reasoning behind the response (Only for reasoning model).
        - `content`: The content of the response.
        - `finish_reason`: The finish reason of the response.
        - `usage`: The usage of the response.
    """
    role: str
    index: Optional[int] = None
    delta_reasoning: Optional[str] = None
    delta_content: Optional[str] = None
    delta_audio: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_results: Optional[List[ToolCallResult]] = None
    tool_notification: Optional[str] = None
    reasoning: Optional[str] = None
    content: Optional[str] = None
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "tool_results", "content_filter", "function_call", "transcription"]] = None
    usage: Optional[UsageResponse] = None
    