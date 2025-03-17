from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union

class OpenAgentResponse(BaseModel):
    """The default Response schema for OpenAgentKit."""
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    refusal: Optional[str] = None
    audio: Optional[Union[str, bytes]] = None