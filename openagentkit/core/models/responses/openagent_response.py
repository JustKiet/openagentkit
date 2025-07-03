from pydantic import BaseModel
from openagentkit.core.models.responses.usage_responses import UsageResponse
from openagentkit.core.models.responses.tool_response import ToolCall
from openagentkit.core.models.responses.audio_response import AudioResponse
from typing import Optional, Any, Union

class OpenAgentResponse(BaseModel):
    """
    The default Response schema for OpenAgentKit.
    
    Schema:
    ```python
    class OpenAgentResponse(BaseModel):
        role: str
        reasoning: Optional[str] = None
        content: Optional[Union[str, BaseModel, dict]] = None
        tool_calls: Optional[List[ToolCall] = None
        tool_results: Optional[List[Any]] = None
        refusal: Optional[str] = None
        audio: Optional[AudioResponse] = None
        usage: Optional[UsageResponse] = None
    ```
    Where:
        - `role`: The role of the response.
        - `reasoning`: The reasoning behind the response (Only for reasoning model).
        - `content`: The content of the response.
        - `tool_calls`: The list of tool calls of the response.
        - `tool_results`: The list of tool results of the response.
        - `refusal`: Response refusal data.
        - `audio`: The audio of the response.
        - `usage`: The usage of the response.
    """
    role: str
    reasoning: Optional[str] = None
    content: Optional[Union[str, BaseModel, dict[str, str]]] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_results: Optional[list[Any]] = None
    refusal: Optional[str] = None
    audio: Optional[AudioResponse] = None
    usage: Optional[UsageResponse] = None