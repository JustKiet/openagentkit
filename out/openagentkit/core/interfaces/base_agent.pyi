import abc
from abc import ABC, abstractmethod
from openagentkit.core.models.responses import OpenAgentResponse as OpenAgentResponse, OpenAgentStreamingResponse as OpenAgentStreamingResponse
from typing import Any, Generator

class BaseAgent(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def clone(self) -> BaseAgent: ...
    @abstractmethod
    def execute(self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None, temperature: float | None = None, max_tokens: int | None = None, top_p: float | None = None) -> Generator[OpenAgentResponse, None, None]: ...
    @abstractmethod
    def stream_execute(self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None, temperature: float | None = None, max_tokens: int | None = None, top_p: float | None = None) -> Generator[OpenAgentStreamingResponse, None, None]: ...
