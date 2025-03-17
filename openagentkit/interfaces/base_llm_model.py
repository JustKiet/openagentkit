from abc import ABC, abstractmethod
from openagentkit.models.responses import OpenAgentResponse
from pydantic import BaseModel
from typing import Union

class BaseLLMModel(ABC):
    @abstractmethod
    def define_system_message(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def model_generate(self) -> Union[OpenAgentResponse, BaseModel]:
        raise NotImplementedError
    