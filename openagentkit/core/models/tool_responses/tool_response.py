from pydantic import BaseModel
from typing import Optional, Union, Any, Literal


class ToolCallResult(BaseModel):
    tool_name: str
    result: Any

class ToolCallMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: Any

# TODO: NOT IMPLEMENTED YET. SHOULD BE USED IN THE FUTURE.
class ToolCallFunction(BaseModel):
    name: str
    arguments: Union[str, dict]

class ToolCallResponse(BaseModel):
    id: str
    type: str
    function: ToolCallFunction

class ToolResponse(BaseModel):
    tool_args: Optional[list[dict]] = None
    tool_calls: Optional[list[dict]] = None
    tool_results: Optional[list[ToolCallResult]] = None
    tool_messages: Optional[list[ToolCallMessage]] = None
    tool_notifications: Optional[list[Union[str, None]]] = None