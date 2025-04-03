from abc import ABC, abstractmethod
from openagentkit.models.responses import OpenAgentResponse
from pydantic import BaseModel
from typing import Union, Optional, Generator, List, Dict, Any

class BaseLLMModel(ABC):
    """
    An abstract base class for LLM models.
    
    ## Methods:
        `model_generate()`: An abstract method to generate a response from the LLM model.

        `model_stream()`: An abstract method to stream a response from the LLM model.

    ## Properties:
        `temperature`: A property to get and set the temperature for the response generation. (defaults to be the range of 0 to 2)

        `max_tokens`: A property to get and set the maximum number of tokens for the response. (defaults to be None)

        `top_p`: A property to get and set the top-p sampling parameter.
    """
    def __init__(self,
                 temperature: Optional[float] = None,
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
        """
        A property to get and set the temperature for the response generation.
        """
        return self._temperature

    @property
    def max_tokens(self) -> int:
        """
        A property to get and set the maximum number of tokens for the response.
        """
        return self._max_tokens
    
    @property
    def top_p(self) -> float:
        """
        A property to get and set the top-p sampling parameter.
        """
        return self._top_p
    
    @temperature.setter
    def temperature(self, value: float) -> None:
        """
        A setter for the temperature property. (defaults to be the range of 0 to 2)
        """
        if value < 0 or value > 2:
            raise ValueError("Temperature must be between 0 and 2")
        self._temperature = value
    
    @top_p.setter
    def top_p(self, value: float) -> None:
        """
        A setter for the top-p sampling parameter property. (defaults to be the range of 0 to 1)
        """
        if value < 0 or value > 1:
            raise ValueError("Top P must be between 0 and 1")
        self._top_p = value
    
    @max_tokens.setter
    def max_tokens(self, value: int) -> None:
        """
        A setter for the maximum number of tokens property.
        """
        self._max_tokens = value

    @abstractmethod
    def model_generate(self,
                       messages: List[Dict[str, str]],
                       response_schema: Optional[BaseModel] = None,
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None,
                       top_p: Optional[float] = None,
                       **kwargs) -> Union[OpenAgentResponse, BaseModel]:
        """
        An abstract method to generate a response from the LLM model.

        Args:
            messages (List[Dict[str, str]]): The messages to be processed.

            response_schema (Optional[BaseModel]): The response schema to be used.

            temperature (Optional[float]): The temperature for the response generation.

            max_tokens (Optional[int]): The maximum number of tokens for the response.

            top_p (Optional[float]): The top-p sampling parameter.

            **kwargs: Additional keyword arguments.

        Returns:
            Union[OpenAgentResponse, BaseModel]: The generated response.
        """
        raise NotImplementedError
    
    @abstractmethod
    def model_stream(self,
                     messages: List[Dict[str, str]],
                     response_schema: Optional[BaseModel] = None,
                     temperature: Optional[float] = None,
                     max_tokens: Optional[int] = None,
                     top_p: Optional[float] = None,
                     **kwargs) -> Generator[OpenAgentResponse, None, None]:
        """
        An abstract method to stream a response from the LLM model.
        
        Args:
            messages (List[Dict[str, str]]): The messages to be processed.

            response_schema (Optional[BaseModel]): The response schema to be used.

            temperature (Optional[float]): The temperature for the response generation.

            max_tokens (Optional[int]): The maximum number of tokens for the response.

            top_p (Optional[float]): The top-p sampling parameter.

            **kwargs: Additional keyword arguments.

        Returns:
            Generator[OpenAgentStreamingResponse, None]: The streamed response.
        """
        raise NotImplementedError