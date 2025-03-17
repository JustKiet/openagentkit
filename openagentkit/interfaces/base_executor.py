from abc import ABC, abstractmethod
from openagentkit.models.responses import OpenAgentResponse
from typing import Optional

class BaseExecutor(ABC):
    @abstractmethod
    def define_system_message(self, system_message: Optional[str]) -> str:
        raise NotImplementedError

    @abstractmethod
    def execute(self) -> OpenAgentResponse:
        raise NotImplementedError