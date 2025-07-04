from pydantic import BaseModel
from typing import Any, Literal

class ToolCallResult(BaseModel):
    tool_name: str
    result: Any

class ToolCallMessage(BaseModel):
    role: Literal['tool']
    tool_call_id: str
    content: Any

class ToolCallFunction(BaseModel):
    name: str
    arguments: str | dict[str, Any]

class ToolCall(BaseModel):
    id: str
    type: str
    function: ToolCallFunction

class ToolResponse(BaseModel):
    tool_args: list[dict[str, Any]] | None
    tool_calls: list[ToolCall] | None
    tool_results: list[ToolCallResult] | None
    tool_messages: list[ToolCallMessage] | None
