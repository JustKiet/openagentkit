import abc
from abc import ABC, abstractmethod
from mcp import ClientSession as ClientSession
from openagentkit.core.models.responses import OpenAgentResponse as OpenAgentResponse, OpenAgentStreamingResponse as OpenAgentStreamingResponse
from typing import Any, AsyncGenerator

class AsyncBaseAgent(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def clone(self) -> AsyncBaseAgent: ...
    @abstractmethod
    async def connect_to_mcp(self, mcp_sessions: list[ClientSession]) -> None: ...
    @abstractmethod
    async def execute(self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None, temperature: float | None = None, max_tokens: int | None = None, top_p: float | None = None) -> AsyncGenerator[OpenAgentResponse, None]: ...
    @abstractmethod
    async def stream_execute(self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None, temperature: float | None = None, max_tokens: int | None = None, top_p: float | None = None) -> AsyncGenerator[OpenAgentStreamingResponse, None]: ...
