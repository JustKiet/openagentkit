from abc import ABC, abstractmethod
from openagentkit.core.models.responses import OpenAgentResponse, OpenAgentStreamingResponse
from typing import Optional, Generator, List, Dict, Any

class BaseAgent(ABC):
    @abstractmethod
    def clone(self) -> 'BaseAgent':
        """
        An abstract method to clone the Agent instance.
        
        Returns:
            BaseAgent: A clone of the Agent instance.
        """
        raise NotImplementedError

    @abstractmethod
    def execute(self,
                messages: List[Dict[str, str]],
                tools: Optional[List[Dict[str, Any]]],
                temperature: Optional[float] = None,
                max_tokens: Optional[int] = None,
                top_p: Optional[float] = None
                ) -> Generator[OpenAgentResponse, None, None]:
        """
        An abstract method to execute an user message with the given tools and parameters.
        
        Args:
            messages (List[Dict[str, str]]): A list of messages to be processed.
            tools (Optional[List[Dict[str, Any]]]): A list of tools to be used.
            temperature (Optional[float]): The temperature for the response generation.
            max_tokens (Optional[int]): The maximum number of tokens for the response.
            top_p (Optional[float]): The top-p sampling parameter.
        Returns:
            Generator[OpenAgentResponse, None, None]: A generator that yields OpenAgentResponse objects.
        """
        raise NotImplementedError
    
    @abstractmethod
    def stream_execute(self,
                       messages: List[Dict[str, str]],
                       tools: Optional[List[Dict[str, Any]]],
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None,
                       top_p: Optional[float] = None
                       ) -> Generator[OpenAgentStreamingResponse, None, None]:
        """
        An abstract method to stream execute an user message with the given tools and parameters."
        
        Args:
            messages (List[Dict[str, str]]): A list of messages to be processed.
            tools (Optional[List[Dict[str, Any]]]): A list of tools to be used.
            temperature (Optional[float]): The temperature for the response generation.
            max_tokens (Optional[int]): The maximum number of tokens for the response.
            top_p (Optional[float]): The top-p sampling parameter.

        Returns:
            Generator[OpenAgentStreamingResponse, None, None]: A generator that yields OpenAgentStreamingResponse objects.
        """
        raise NotImplementedError
