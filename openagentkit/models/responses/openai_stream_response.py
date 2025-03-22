from pydantic import BaseModel
from typing import Literal, Optional, List, Union, Dict, Any
from openagentkit.models.responses.usage_responses import UsageResponse

class OpenAIStreamingResponse(BaseModel):
    role: str
    index: Optional[int] = None
    delta_content: Optional[str] = None
    tool_calls: Optional[List[Union[Dict[str, Any], BaseModel]]] = None
    tool_notification: Optional[str] = None
    content: Optional[str] = None
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter"]] = None
    usage: Optional[UsageResponse] = None
    