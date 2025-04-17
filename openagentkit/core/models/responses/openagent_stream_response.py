from pydantic import BaseModel
from typing import Literal, Optional, List, Union, Dict, Any
from openagentkit.core.models.responses.usage_responses import UsageResponse

class OpenAgentStreamingResponse(BaseModel):
    """
    The response schema for OpenAgentKit streaming responses.

    Schema:
        ```python
        class OpenAgentStreamingResponse(BaseModel):
            role: str
            index: Optional[int] = None
            delta_content: Optional[str] = None
            delta_audio: Optional[str] = None
            tool_calls: Optional[List[Union[Dict[str, Any], BaseModel]]] = None
            tool_notification: Optional[str] = None
            content: Optional[str] = None
            finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter"]] = None
            usage: Optional[UsageResponse] = None
        ```
    Where:
        - `role`: The role of the response.
        - `index`: The index of the response.
        - `delta_content`: The delta content of the response.
        - `delta_audio`: The delta audio in base64 format of the response.
        - `tool_calls`: The tool calls of the response.
        - `tool_notification`: The tool notification of the response.
        - `content`: The content of the response.
        - `finish_reason`: The finish reason of the response.
        - `usage`: The usage of the response.
    """
    role: str
    index: Optional[int] = None
    delta_content: Optional[str] = None
    delta_audio: Optional[str] = None
    tool_calls: Optional[List[Union[Dict[str, Any], BaseModel, Any]]] = None
    tool_notification: Optional[str] = None
    content: Optional[str] = None
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter"]] = None
    usage: Optional[UsageResponse] = None
    