from pydantic import BaseModel
from openagentkit.models.responses.usage_responses import UsageResponse
from typing import Optional, List, Dict, Any, Union

class OpenAgentResponse(BaseModel):
    """The default Response schema for OpenAgentKit."""
    role: str
    content: Optional[Union[str, BaseModel, dict]] = None
    tool_calls: Optional[List[Union[Dict[str, Any], BaseModel]]] = None
    tool_results: Optional[List[Union[Dict[str, Any], BaseModel]]] = None
    tool_notification: Optional[str] = None
    refusal: Optional[str] = None
    audio: Optional[Union[str, bytes]] = None
    usage: Optional[UsageResponse] = None