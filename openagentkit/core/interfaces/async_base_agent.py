from abc import ABC, abstractmethod
from openagentkit.core.models.responses import OpenAgentResponse, OpenAgentStreamingResponse
from typing import Optional, AsyncGenerator, List, Dict, Any
from mcp import ClientSession

class AsyncBaseAgent(ABC):
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
    