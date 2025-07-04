from pydantic import BaseModel, Field
from typing import Any

class ContextUnit(BaseModel):
    thread_id: str
    agent_id: str
    history: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])
    created_at: int
    updated_at: int

