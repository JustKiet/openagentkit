from typing import List, Optional, Callable, Any
from loguru import logger
from openai._types import NOT_GIVEN
import json
from pydantic import BaseModel
from openagentkit.models.responses import OpenAgentResponse, OpenAIStreamingResponse
from openagentkit.models.tool_responses import ToolResponse

class ToolHandler:
    def __init__(self,
                 tools: Optional[List[Callable[..., Any]]] = NOT_GIVEN,
                 *args,
                 **kwargs):
        self._tools = [
            tool.schema for tool in tools
        ] if tools else NOT_GIVEN
        self.tools_map = {
            tool.schema["function"]["name"]: tool for tool in tools
        } if tools is not NOT_GIVEN else NOT_GIVEN
        
    @property
    def tools(self):
        return self._tools
    
    @tools.setter
    def tools(self, tools):
        self._tools = [tool.schema for tool in tools] if tools else NOT_GIVEN
        self.tools_map = {
            tool.schema["function"]["name"]: tool for tool in tools
        } if tools is not NOT_GIVEN else NOT_GIVEN
        return logger.info(f"Binded {len(self._tools)} tools.")
    
    def _handle_tool_call(self, tool_name, **kwargs) -> BaseModel:
        if self.tools_map is not NOT_GIVEN:
            tool = self.tools_map.get(tool_name)
            if not tool:
                return None
            return tool(**kwargs)
        else:
            logger.error("No tools provided")
            return None
    
    def parse_tool_args(self, response: dict):
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
    
    def handle_tool_request(self, response: OpenAgentResponse) -> ToolResponse:
        """
        Handle tool requests and get the final response with tool results
        """
        assert type(response) == OpenAgentResponse or type(response) == OpenAIStreamingResponse, "Response must be an OpenAgentResponse or OpenAIStreamingResponse object"
        
        tool_args_list = []
        tool_results_list = []
        tool_messages_list = []
        notifications_list = []
        
        # Check if the response contains tool calls
        if response.tool_calls is None:
            logger.debug("No tool calls found in the response. Skipping tool call handling.")
            return tool_results_list, tool_messages_list

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
            tool_results_list.append({
                "tool_call": tool_call,
                "result": tool_result,
            })
            
            logger.info(f"Tool Result: {tool_result}")
            
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
    