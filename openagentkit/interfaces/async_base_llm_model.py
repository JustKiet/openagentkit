from abc import ABC, abstractmethod
from openagentkit.models.responses import OpenAgentResponse
from pydantic import BaseModel
from typing import Union

class AsyncBaseLLMModel(ABC):
    @abstractmethod
    async def define_system_message(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    async def model_generate(self) -> Union[OpenAgentResponse, BaseModel]:
        raise NotImplementedError
    