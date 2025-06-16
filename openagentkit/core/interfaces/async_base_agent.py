from abc import ABC, abstractmethod
from openagentkit.core.models.responses import OpenAgentResponse, OpenAgentStreamingResponse
from typing import Optional, AsyncGenerator, List, Dict, Any
from mcp import ClientSession

class AsyncBaseAgent(ABC):
    def __init__(self,
                 system_message: Optional[str] = None, 
                 context_history: Optional[List[Dict[str, Any]]] = None):
        self._system_message = system_message or "You are a helpful assistant. Try to assist the user as best as you can. If you are unsure, ask clarifying questions. If you don't know the answer, say 'I don't know'."

        self._context_history: List[Dict[str,  Any]] = [
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
    def system_message(self, value: str) -> None:
        """
        Set the system message.

        Args:
            value: The system message to set.
        """
        self._system_message = value
        self._context_history[0]["content"] = value

    @abstractmethod
    def clone(self) -> 'AsyncBaseAgent':
        """
        An abstract method to clone the Agent instance.
        
        Returns:
            AsyncBaseAgent: A clone of the Agent instance.
        """
        pass
    
    @abstractmethod
    async def connect_to_mcp(self, mcp_sessions: List[ClientSession]) -> None:
        """
        An abstract method to connect the Agent to the MCP sessions.
        
        Args:
            sessions (List[Dict[str, Any]]): The sessions to be connected.
        """
        pass

    @abstractmethod
    async def execute(self,
                      messages: List[Dict[str, str]],
                      tools: Optional[List[Dict[str, Any]]] = None,
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
            OpenAgentResponse: The response from the Agent.
        """
        while False:
            yield
        pass
    
    @abstractmethod
    async def stream_execute(self,
                             messages: List[Dict[str, str]],
                             tools: Optional[List[Dict[str, Any]]] = None,
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
        while False:
            yield
        pass
    
    async def get_history(self) -> List[Dict[str, Any]]:
        """
        An abstract method to get the history of the conversation.

        Returns:
            List[Dict[str, Any]]: The history of the conversation.
        """
        return self._context_history
    
    async def add_context(self, content: dict[str, Any]):
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
    
    async def extend_context(self, content: List[dict[str, Any]]):
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
    
    async def clear_context(self):
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