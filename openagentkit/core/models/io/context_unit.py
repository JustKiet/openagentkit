from pydantic import BaseModel, Field


class ContextUnit(BaseModel):
    thread_id: str
    agent_id: str
    history: list[dict[str, str]] = Field(default_factory=list[dict[str, str]])
    created_at: int
    updated_at: int

