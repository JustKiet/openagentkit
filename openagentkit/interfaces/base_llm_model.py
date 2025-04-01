from abc import ABC, abstractmethod
from openagentkit.models.responses import OpenAgentResponse
from pydantic import BaseModel
from typing import Union, Optional, Generator

class BaseLLMModel(ABC):
    def __init__(self,
                 temperature: Optional[float] = 0.3,
                 max_tokens: Optional[int] = None,
                 top_p: Optional[float] = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def max_tokens(self) -> int:
        return self._max_tokens
    
    @property
    def top_p(self) -> float:
        return self._top_p
    
    @temperature.setter
    def temperature(self, value: float) -> None:
        if value < 0 or value > 2:
            raise ValueError("Temperature must be between 0 and 2")
        self._temperature = value
    
    @top_p.setter
    def top_p(self, value: float) -> None:
        if value < 0 or value > 1:
            raise ValueError("Top P must be between 0 and 1")
        self._top_p = value
    
    @max_tokens.setter
    def max_tokens(self, value: int) -> None:
        self._max_tokens = value

    @abstractmethod
    def model_generate(self) -> Union[OpenAgentResponse, BaseModel]:
        raise NotImplementedError
    
    @abstractmethod
    def model_stream(self) -> Generator[OpenAgentResponse, None, None]:
        raise NotImplementedError
    
    