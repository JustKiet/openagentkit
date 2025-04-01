from abc import ABC, abstractmethod
from openagentkit.models.responses import OpenAgentResponse, OpenAgentStreamingResponse
from typing import Optional, Generator

class BaseExecutor(ABC):
    @abstractmethod
    def define_system_message(self, system_message: Optional[str]) -> str:
        raise NotImplementedError

    @abstractmethod
    def execute(self) -> OpenAgentResponse:
        raise NotImplementedError
    
    @abstractmethod
    def stream_execute(self) -> Generator[OpenAgentStreamingResponse, None, None]:
        raise NotImplementedError