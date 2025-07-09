from pydantic import BaseModel, Field
from typing import Any

class ContextUnit(BaseModel):
    thread_id: str
    agent_id: str
    history: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])
    created_at: int
    updated_at: int

    @property
    def system_message(self) -> str:
        """
        Get the system message for the context unit.

        Returns:
            The system message.
        """
        return self.history[0].get("system_message", "")