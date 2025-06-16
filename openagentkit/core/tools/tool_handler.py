from typing import List, Optional, Any, Union
import json
from openagentkit.core.exceptions import InvalidToolSchemaError
from openagentkit.core.models.responses import ToolResponse, ToolCallResult, ToolCallMessage, ToolCall, ToolCallFunction
from openagentkit.core.tools.base_tool import Tool
from mcp import ClientSession
from mcp.types import CallToolResult
import inspect
import logging

logger = logging.getLogger(__name__)

class ToolHandler:
    def __init__(self,
                 tools: Optional[List[Tool]] = None,
                 mcp_sessions: Optional[dict[str, ClientSession]] = None,
                 mcp_tools: Optional[dict[str, list[str]]] = None,
                 ):
        self._tools: Optional[list[dict[str, Any]]] = None
        self.tools_map: Optional[dict[str, Union[Tool, dict[str, str]]]] = None

        if tools:
            self._tools = []
            for tool in tools:
                if not hasattr(tool, "schema"):
                    raise ValueError(f"Function '{tool.__name__}' does not have a `schema` attribute. Please wrap the function with `@tool` decorator from `openagentkit.core.utils.tool_wrapper`.")
                self._tools.append(tool.schema)
                self.tools_map = {
                    tool.schema["function"]["name"]: tool for tool in tools
                }

        self.sessions_map = mcp_sessions
        self.mcp_tools_map = mcp_tools


    @classmethod
    async def from_mcp(
        cls, 
        sessions: list[ClientSession],
        additional_tools: Optional[List[Tool]] = None
    ) -> "ToolHandler":
        mcp_sessions: Optional[dict[str, ClientSession]] = {}
        mcp_tools: Optional[dict[str, list[str]]] = {}
        for session in sessions:
            initialization = await session.initialize()

            mcp_sessions[initialization.serverInfo.name] = session

            list_tools = await session.list_tools()
            tool_names = [tool.name for tool in list_tools.tools]
            mcp_tools[initialization.serverInfo.name] = tool_names

        self = cls(mcp_sessions=mcp_sessions, mcp_tools=mcp_tools, tools=additional_tools)

        for session in sessions:
            await self.load_mcp_tools(session=session)

        return self
    
    def _handle_mcp_tool_schema(self, tool: dict[str, Any]) -> dict[str, Any]:
        tool["parameters"] = tool.pop("inputSchema")
        return tool

    async def load_mcp_tools(self, session: ClientSession):
        tool_list = await session.list_tools()

        tool_schemas = [tool.model_dump() for tool in tool_list.tools]
        
        parsed_tools: list[dict[str, Any]] = []

        for tool in tool_schemas:
            # Check if the tool is already in the tools list
            tool = self._handle_mcp_tool_schema(tool)
            parsed_tools.append(tool)
        
        mcp_tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": tool
            }
            for tool in parsed_tools
        ]

        if self._tools is None:
            self._tools = []
        if self.tools_map is None:
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
    def tools(self, tools: List[Tool]):
        self._tools = [tool.schema for tool in tools]
        self.tools_map: Optional[dict[str, Union[Tool, dict[str, str]]]] = {
            tool.schema["function"]["name"]: tool for tool in tools
        }

    async def _handle_mcp_tool_call(self, tool_name: str, **kwargs: Any) -> Any:
        tool_arguments = kwargs

        if self.mcp_tools_map is None:
            logger.error("No MCP tools provided")
            return None
        
        for session_name, tools in self.mcp_tools_map.items():
            if self.sessions_map is None:
                logger.error("No MCP sessions provided")
                return None
            
            if tool_name in tools and self.sessions_map.get(session_name, None) is not None:
                tool_results: CallToolResult = await self.sessions_map[session_name].call_tool(
                    name=tool_name, arguments=tool_arguments,
                )
                return str([tool_result.model_dump() for tool_result in tool_results.content])
            
    async def _async_handle_tool_call(self, tool_name: str, **kwargs: Any) -> Any:
        if self.tools_map is None:
            logger.error("No tools provided")
            return None

        tool = self.tools_map.get(tool_name)
        if not tool:
            return None

        if callable(tool):
            result = tool(**kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        # If it's not callable, fallback to MCP tool handler
        return await self._handle_mcp_tool_call(tool_name, **kwargs)
        
    def _handle_tool_call(self, tool_name: str, **kwargs: Any) -> Any:
        if self.tools_map is not None:
            tool = self.tools_map.get(tool_name, None)
            if not tool:
                return None
            elif callable(tool):
                return tool(**kwargs)
        else:
            logger.error("No tools provided")
            return None
        
    def parse_tools_args(self, tool_calls_dict: list[dict[str, Any]]) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []

        try:
            for tool_call in tool_calls_dict:
                ToolCall(
                    id=tool_call["id"],
                    type=tool_call["type"],
                    function=ToolCallFunction(
                        name=tool_call["function"]["name"],
                        arguments=tool_call["function"]["arguments"],
                    ),
                )
            return tool_calls

        except KeyError as e:
            raise InvalidToolSchemaError() from e

    async def async_handle_tool_request(self, tool_calls: list[ToolCall]) -> ToolResponse:   
        tool_args_list: list[dict[str, Any]] = []
        tool_results_list: list[ToolCallResult] = []
        tool_messages_list: list[ToolCallMessage] = []
        notifications_list: list[str] = []

        # Handle tool calls 
        for tool_call in tool_calls:
            tool_call_id = tool_call.id
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse tool arguments for {tool_name}: {e}")
                    tool_args = {}
            
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
            
            # Convert tool result to string if it's not already a string
            tool_result_str = str(tool_result)

            tool_messages_list.append(
                ToolCallMessage(
                    role="tool",
                    tool_call_id=tool_call_id,
                    content=tool_result_str
                )
            )
        
        return ToolResponse(
            tool_args=tool_args_list,
            tool_calls=tool_calls,
            tool_results=tool_results_list,
            tool_messages=tool_messages_list,
            tool_notifications=notifications_list
        )
    
    def handle_tool_request(self, tool_calls: list[ToolCall]) -> ToolResponse:
        tool_args_list: list[dict[str, Any]] = []
        tool_results_list: list[ToolCallResult] = []
        tool_messages_list: list[ToolCallMessage] = []
        notifications_list: list[str] = []

        # Handle tool calls 
        for tool_call in tool_calls:
            tool_call_id = tool_call.id
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse tool arguments for {tool_name}: {e}")
                    tool_args = {}
            # Save notification value and remove _notification key from tool args if present
            notification: Optional[str] = tool_args.get("_notification", None)

            if notification:
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
            
            # Convert tool result to string if it's not already a string
            tool_result_str = str(tool_result)

            tool_messages_list.append(
                ToolCallMessage(
                    role="tool",
                    tool_call_id=tool_call_id,
                    content=tool_result_str
                )
            )
        
        return ToolResponse(
            tool_args=tool_args_list,
            tool_calls=tool_calls,
            tool_results=tool_results_list,
            tool_messages=tool_messages_list,
            tool_notifications=notifications_list
        )
    