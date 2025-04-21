from typing import List, Optional, Callable, Any, Union
from loguru import logger
from openai._types import NOT_GIVEN
import json
from openagentkit.core.models.responses import OpenAgentResponse, OpenAgentStreamingResponse
from openagentkit.core.models.tool_responses import ToolResponse, ToolCallResult
from openagentkit.core.interfaces import BaseToolHandler
from mcp import ClientSession

class ToolHandler(BaseToolHandler):
    """
    A class to handle tool calls.
    
    ## Methods:
        `parse_tool_args()`: A method to parse the tool calls from the response.

        `handle_notification()`: A method to handle the notification from the tool call chunk.

        `handle_tool_request()`: A method to handle the tool request and get the final response with tool results.

    ## Properties:
        `tools`: A property to get and set the tools.

        `tools_map`: A property to get the tools map.
    """
    def __init__(self,
                 tools: Optional[List[Callable[..., Any]]] = NOT_GIVEN,
                 mcp_session: ClientSession = None,
                 *args,
                 **kwargs):
        
        self._tools = NOT_GIVEN
        self.tools_map = NOT_GIVEN

        if tools is not NOT_GIVEN:
            self._tools = [
                tool.schema for tool in tools
            ]
            self.tools_map = {
                tool.schema["function"]["name"]: tool for tool in tools
            }

        self.session: Optional[ClientSession] = mcp_session

    @classmethod
    async def from_mcp(cls, session: ClientSession, additional_tools: Optional[List[Callable[..., Any]]] = NOT_GIVEN):
        """Asynchronous factory method to create an instance with tools loaded."""
        self = cls(mcp_session=session, tools=additional_tools)
        await self.load_mcp_tools()
        return self

    async def load_mcp_tools(self):
        tool_list = await self.session.list_tools()

        tool_schemas = [tool.model_dump() for tool in tool_list.tools]
        
        parsed_tools = []

        for tool in tool_schemas:
            # Check if the tool is already in the tools list
            tool["parameters"] = tool.pop("inputSchema")
            parsed_tools.append(tool)
        
        mcp_tools = [
            {
                "type": "function",
                "function": tool
            }
            for tool in parsed_tools
        ]

        if self._tools is NOT_GIVEN:
            self._tools = []
        if self.tools_map is NOT_GIVEN:
            self.tools_map = {}
        # Extend the existing tools with the loaded MCP tools
        self._tools.extend(mcp_tools)

        # Update the tools map with the loaded MCP tools
        self.tools_map.update({
            tool["function"]["name"]: tool for tool in mcp_tools
        })
        
    @property
    def tools(self):
        return self._tools
    
    @tools.setter
    def tools(self, tools):
        self._tools = [tool.schema for tool in tools] if tools else NOT_GIVEN
        self.tools_map = {
            tool.schema["function"]["name"]: tool for tool in tools
        } if tools is not NOT_GIVEN else NOT_GIVEN
        return f"Binded {len(self._tools)} tools."
    
    async def _async_handle_tool_call(self, tool_name: str, **kwargs) -> Any:
        """
        Handle the tool call asynchronously and return the tool result.

        Args:
            tool_name (str): The name of the tool to handle.
            **kwargs: The keyword arguments to pass to the tool.

        Returns:
            Any: The result of the tool call.
        """
        if self.tools_map is not NOT_GIVEN:
            tool = self.tools_map.get(tool_name, None)
            if not tool:
                return None
            elif callable(tool):
                return tool(**kwargs)
            elif isinstance(tool, dict):
                tool_arguments = kwargs
                tool_results = await self.session.call_tool(
                    name=tool_name, arguments=tool_arguments,
                )
                return str([tool_result.model_dump() for tool_result in tool_results.content])
        else:
            logger.error("No tools provided")
            return None
        
    def _handle_tool_call(self, tool_name: str, **kwargs) -> Any:
        """
        Handle the tool call and return the tool result.

        Args:
            tool_name (str): The name of the tool to handle.
            **kwargs: The keyword arguments to pass to the tool
        
        Returns:
            Any: The result of the tool call.
        """
        if self.tools_map is not NOT_GIVEN:
            tool = self.tools_map.get(tool_name, None)
            if not tool:
                return None
            elif callable(tool):
                return tool(**kwargs)
        else:
            logger.error("No tools provided")
            return None
    
    def parse_tool_args(self, response: dict) -> list[dict[str, Any]]:
        """
        Parse the tool calls from the response.

        Args:
            response (dict): The response from the OpenAI model.

        Returns:
            list[dict[str, Any]]: The tool calls.
        """
        tool_calls = None
        if hasattr(response, "tool_calls") and response.tool_calls is not None:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "arguments": tc.function.arguments,
                        "name": tc.function.name,
                    },
                }
                for tc in response.tool_calls
            ]

        return tool_calls
    
    def handle_notification(self, chunk: OpenAgentStreamingResponse) -> Union[OpenAgentStreamingResponse, None]:
        """
        Handle the notification from the tool call chunk

        Args:
            chunk (OpenAgentStreamingResponse): An OpenAgentStreamingResponse object.

        Returns:
            Union[OpenAgentStreamingResponse, None]: The notification.
        """
        notification = chunk.tool_calls[0].get("function")
        tool_notification = None
        
        if notification.get("arguments"):
            if type(notification.get("arguments")) == str:
                args = json.loads(notification.get("arguments"))
            else:
                args = notification.get("arguments")

            if args.get("_notification"):
                tool_notification = args.get("_notification", None)

            if notification:
                #logger.info(f"Tool Notification: {tool_notification}")
                return OpenAgentStreamingResponse(
                    role="assistant",
                    content=None,
                    tool_notification=tool_notification,
                )
            
        return None

    async def async_handle_tool_request(self, response: Union[OpenAgentResponse, OpenAgentStreamingResponse]) -> ToolResponse:
        """
        Handle tool requests and get the final response with tool results

        Args:
            response (OpenAgentResponse or OpenAgentStreamingResponse): The response from the OpenAI model.

        Returns:
            ToolResponse: The final response with tool results.
        """
        if type(response) != OpenAgentResponse and type(response) != OpenAgentStreamingResponse:
            raise AttributeError("Response must be an OpenAgentResponse or OpenAgentStreamingResponse object")
        
        tool_args_list = []
        tool_results_list = []
        tool_messages_list = []
        notifications_list = []
        
        # Check if the response contains tool calls
        if response.tool_calls is None:
            #logger.debug("No tool calls found in the response. Skipping tool call handling.")
            return ToolResponse(
                tool_args=[],
                tool_calls=[],
                tool_results=[],
                tool_messages=[],
                tool_notifications=[]
            )

        # Handle tool calls 
        for tool_call in response.tool_calls:
            tool_call_id = tool_call.get("id")
            tool_name = tool_call.get("function").get("name")
            tool_args: dict = eval(tool_call.get("function").get("arguments"))
            # Save notification value and remove _notification key from tool args if present
            notification = tool_args.get("_notification", None)
            notifications_list.append(notification)
            tool_args.pop("_notification", None)
            
            # Handle the tool call (execute the tool)
            tool_result = await self._async_handle_tool_call(tool_name, **tool_args)
            
            # Store the tool args
            tool_args_list.append(tool_args)

            # Store tool call and result
            tool_results_list.append(
                ToolCallResult(
                    tool_name=tool_name,
                    result=tool_result
                )
            )
            
            #logger.info(f"Tool Result: {tool_result}")
            
            # Convert tool result to string if it's not already a string
            tool_result_str = str(tool_result)
            
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": tool_result_str,  # Use string representation
            }

            tool_messages_list.append(tool_message)  
        
        return ToolResponse(
            tool_args=tool_args_list,
            tool_calls=response.tool_calls,
            tool_results=tool_results_list,
            tool_messages=tool_messages_list,
            tool_notifications=notifications_list
        )
    
    def handle_tool_request(self, response: Union[OpenAgentResponse, OpenAgentStreamingResponse]) -> ToolResponse:
        """
        Handle tool requests and get the final response with tool results

        Args:
            response (OpenAgentResponse or OpenAgentStreamingResponse): The response from the OpenAI model.

        Returns:
            ToolResponse: The final response with tool results.
        """
        if type(response) != OpenAgentResponse and type(response) != OpenAgentStreamingResponse:
            raise AttributeError("Response must be an OpenAgentResponse or OpenAgentStreamingResponse object")
        
        tool_args_list = []
        tool_results_list = []
        tool_messages_list = []
        notifications_list = []
        
        # Check if the response contains tool calls
        if response.tool_calls is None:
            #logger.debug("No tool calls found in the response. Skipping tool call handling.")
            return ToolResponse(
                tool_args=[],
                tool_calls=[],
                tool_results=[],
                tool_messages=[],
                tool_notifications=[]
            )

        # Handle tool calls 
        for tool_call in response.tool_calls:
            tool_call_id = tool_call.get("id")
            tool_name = tool_call.get("function").get("name")
            tool_args: dict = eval(tool_call.get("function").get("arguments"))
            # Save notification value and remove _notification key from tool args if present
            notification = tool_args.get("_notification", None)
            notifications_list.append(notification)
            tool_args.pop("_notification", None)
            
            # Handle the tool call (execute the tool)
            tool_result = self._handle_tool_call(tool_name, **tool_args)
            
            # Store the tool args
            tool_args_list.append(tool_args)

            # Store tool call and result
            tool_results_list.append(
                ToolCallResult(
                    tool_name=tool_name,
                    result=tool_result
                )
            )
            
            #logger.info(f"Tool Result: {tool_result}")
            
            # Convert tool result to string if it's not already a string
            tool_result_str = str(tool_result)
            
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": tool_result_str,  # Use string representation
            }

            tool_messages_list.append(tool_message)  
        
        return ToolResponse(
            tool_args=tool_args_list,
            tool_calls=response.tool_calls,
            tool_results=tool_results_list,
            tool_messages=tool_messages_list,
            tool_notifications=notifications_list
        )
    