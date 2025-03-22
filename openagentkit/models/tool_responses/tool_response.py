from pydantic import BaseModel

class ToolCallFunction(BaseModel):
    name: str
    arguments: str

class ToolCallResponse(BaseModel):
    id: str
    type: str
    function: ToolCallFunction

class ToolResultResponse(BaseModel):
    tool_call: ToolCallResponse
    result: dict