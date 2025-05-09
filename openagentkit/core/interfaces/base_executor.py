from abc import ABC, abstractmethod
from openagentkit.core.models.responses import OpenAgentResponse, OpenAgentStreamingResponse
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
    def __init__(self,
                 system_message: Optional[str] = None, 
                 context_history: Optional[List[Dict[str, str]]] = None):
        self._system_message = system_message or "You are a helpful assistant. Try to assist the user as best as you can. If you are unsure, ask clarifying questions. If you don't know the answer, say 'I don't know'."

        self._context_history = [
            {
                "role": "system",
                "content": self._system_message,
            }
        ]

        if context_history is not None:
            self._context_history = context_history

    @property
    def system_message(self) -> str:
        """
        Get the system message.

        Returns:
            The system message.
        """
        return self._system_message
    
    @system_message.setter
    def system_message(self, value: str):
        """
        Set the system message.

        Args:
            value: The system message to set.
        """
        self._system_message = value
        self._context_history[0]["content"] = value

    @abstractmethod
    def clone(self) -> 'BaseExecutor':
        """
        An abstract method to clone the executor instance.
        
        Returns:
            BaseExecutor: A clone of the executor instance.
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
    
    def get_context(self) -> List[Dict[str, Any]]:
        """
        An abstract method to get the history of the conversation.

        Returns:
            List[Dict[str, Any]]: The history of the conversation.
        """
        return self._context_history
    
    def add_context(self, content: dict[str, str]):
        """
        Add context to the model.

        Args:
            content: The content to add to the context.

        Returns:
            The context history.
        """
        if not content:
            return self._context_history
        
        self._context_history.append(content)
        return self._context_history
    
    def extend_context(self, content: List[dict[str, str]]):
        """
        Extend the context of the model.

        Args:
            content: The content to extend the context with.

        Returns:
            The context history.
        """
        if not content:
            return self._context_history
        
        self._context_history.extend(content)
        return self._context_history
    
    def clear_context(self):
        """
        Clear the context of the model leaving only the system message.

        Returns:
            The cleared context history.
        """
        self._context_history = [
            {
                "role": "system",
                "content": self._system_message,
            }
        ]
        return self._context_history