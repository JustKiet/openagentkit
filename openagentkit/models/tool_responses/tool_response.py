from pydantic import BaseModel
from typing import Optional, Union

# TODO: NOT IMPLEMENTED YET. SHOULD BE USED IN THE FUTURE.
class ToolCallFunction(BaseModel):
    name: str
    arguments: str

class ToolCallResponse(BaseModel):
    id: str
    type: str
    function: ToolCallFunction

class ToolResponse(BaseModel):
    tool_args: Optional[list[dict]] = None
    tool_calls: Optional[list[dict]] = None
    tool_results: Optional[list[dict]] = None
    tool_messages: Optional[list[dict]] = None
    tool_notifications: Optional[list[Union[str, None]]] = None