from abc import ABC, abstractmethod
from openagentkit.core.models.responses import OpenAgentResponse, OpenAgentStreamingResponse
from typing import Optional, AsyncGenerator, List, Dict, Any

class AsyncBaseExecutor(ABC):
    """
    An abstract base class for executing user messages with tools and parameters.
    This class defines the interface for executing messages and provides methods
    for defining system messages and executing user messages with tools.
    It is intended to be subclassed by concrete implementations that provide
    specific execution logic.
    
    ## Methods:
        `define_system_message()`: An abstract method to define the system message for the executor.

        `execute()`: An abstract method to execute a user message with the given tools and parameters.

        `stream_execute()`: An abstract method to stream execute a user message with the given tools and parameters.
    """

    @abstractmethod
    def clone(self) -> 'AsyncBaseExecutor':
        """
        An abstract method to clone the executor instance.
        
        Returns:
            AsyncBaseExecutor: A clone of the executor instance.
        """
        raise NotImplementedError

    @abstractmethod
    async def define_system_message(self, system_message: Optional[str]) -> str:
        """
        An abstract method to define the system message for the executor.
        
        Args:
            system_message (Optional[str]): The system message to be defined.
        
        Returns:
            str: The defined system message.
        """
        raise NotImplementedError

    @abstractmethod
    async def execute(self,
                      messages: List[Dict[str, str]],
                      tools: Optional[List[Dict[str, Any]]],
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      top_p: Optional[float] = None) -> AsyncGenerator[OpenAgentResponse, None]:
        """
        An abstract method to execute a user message with the given tools and parameters.
        
        Args:
            messages (List[Dict[str, str]]): The messages to be processed.

            tools (Optional[List[Dict[str, Any]]]): The tools to be used.

            temperature (Optional[float]): The temperature for the response generation.

            max_tokens (Optional[int]): The maximum number of tokens for the response.

            top_p (Optional[float]): The top-p sampling parameter.

        Returns:
            OpenAgentResponse: The response from the executor.
        """
        raise NotImplementedError
    
    @abstractmethod
    async def stream_execute(self,
                             messages: List[Dict[str, str]],
                             tools: Optional[List[Dict[str, Any]]],
                             temperature: Optional[float] = None,
                             max_tokens: Optional[int] = None,
                             top_p: Optional[float] = None) -> AsyncGenerator[OpenAgentStreamingResponse, None]:
        """
        An abstract method to stream execute a user message with the given tools and parameters.
        
        Args:
            messages (List[Dict[str, str]]): The messages to be processed.

            tools (Optional[List[Dict[str, Any]]]): The tools to be used.

            temperature (Optional[float]): The temperature for the response generation.

            max_tokens (Optional[int]): The maximum number of tokens for the response.

            top_p (Optional[float]): The top-p sampling parameter.

        Returns:
            AsyncGenerator[OpenAgentStreamingResponse, None]: The streamed response.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_history(self) -> List[Dict[str, Any]]:
        """
        An abstract method to get the history of the conversation.

        Returns:
            List[Dict[str, Any]]: The history of the conversation.
        """
        raise NotImplementedError