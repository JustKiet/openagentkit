from abc import ABC, abstractmethod
from openagentkit.core.models.responses import OpenAgentStreamingResponse, OpenAgentResponse
from openagentkit.core.models.responses.tool_response import ToolResponse, ToolCallResponse
from typing import Union

class BaseToolHandler(ABC):
    @abstractmethod
    def parse_tool_args(self, response: OpenAgentResponse) -> list[ToolCallResponse]:
        """
        Parse the tool calls from the response.

        Args:
            response (dict): An OpenAgentStreamingResponse object.

        Returns:
            list[dict[str, str]]: The tool calls.
        """
        raise NotImplementedError
    
    @abstractmethod
    def handle_notification(self, chunk: OpenAgentStreamingResponse) -> Union[OpenAgentStreamingResponse, None]:
        """
        Handle the notification from the tool call chunk

        Args:
            chunk (dict): The chunk from the OpenAI model.

        Returns:
            Union[OpenAgentStreamingResponse, None]: The notification.
        """
        raise NotImplementedError
    
    @abstractmethod
    def handle_tool_request(self, response: Union[OpenAgentResponse, OpenAgentStreamingResponse]) -> ToolResponse:
        """
        Handle tool requests and get the final response with tool results

        Args:
            response (OpenAgentResponse or OpenAgentStreamingResponse): The response from the OpenAI model.

        Returns:
            ToolResponse: The final response with tool results.
        """
        raise NotImplementedError 