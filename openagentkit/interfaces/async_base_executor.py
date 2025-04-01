from abc import ABC, abstractmethod
from openagentkit.models.responses import OpenAgentResponse, OpenAgentStreamingResponse
from typing import Optional, AsyncGenerator

class AsyncBaseExecutor(ABC):
    @abstractmethod
    async def define_system_message(self, system_message: Optional[str]) -> str:
        raise NotImplementedError

    @abstractmethod
    async def execute(self) -> OpenAgentResponse:
        raise NotImplementedError
    
    @abstractmethod
    async def stream_execute(self) -> AsyncGenerator[OpenAgentStreamingResponse, None]:
        raise NotImplementedError