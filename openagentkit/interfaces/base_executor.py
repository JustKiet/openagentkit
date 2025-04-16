from abc import ABC, abstractmethod
from openagentkit.models.responses import OpenAgentResponse, OpenAgentStreamingResponse
from typing import Optional, Generator, List, Dict, Any

class BaseExecutor(ABC):
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
    def clone(self) -> 'BaseExecutor':
        """
        An abstract method to clone the executor instance.
        
        Returns:
            BaseExecutor: A clone of the executor instance.
        """
        raise NotImplementedError

    @abstractmethod
    def define_system_message(self, system_message: Optional[str]) -> str:
        """
        An abstract method to define the system message for the executor.

        Args:
            system_message (Optional[str]): The system message to be defined.
        
        Returns:
            str: The defined system message.
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
    
    @abstractmethod
    def get_history(self) -> List[Dict[str, Any]]:
        """
        An abstract method to get the history of the conversation.

        Returns:
            List[Dict[str, Any]]: The history of the conversation.
        """
        raise NotImplementedError