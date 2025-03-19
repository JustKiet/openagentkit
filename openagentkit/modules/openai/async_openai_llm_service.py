from typing import Any, Callable, Dict, List, Optional, Union
from openai import AsyncOpenAI
from openai._types import NOT_GIVEN
from pydantic import BaseModel
from openagentkit.handlers.tool_handler import ToolHandler
from openagentkit.interfaces import AsyncBaseLLMModel
from openagentkit.models import OpenAgentResponse
import os

class AsyncOpenAILLMService(AsyncBaseLLMModel):
    def __init__(self, 
                 client: AsyncOpenAI,
                 model: str = "gpt-4o-mini",
                 system_message: Optional[str] = None,
                 tools: Optional[List[Callable[..., Any]]] = NOT_GIVEN,
                 api_key: Optional[str] = os.getenv("OPENAI_API_KEY"),
                 temperature: Optional[float] = 0.3,
                 max_tokens: Optional[int] = None,
                 top_p: Optional[float] = None,
                 *args,
                 **kwargs):
        # Create an instance of ToolHandler instead of inheriting from it
        self._tool_handler = ToolHandler(tools=tools)
        
        self._client = client
        self._model = model
        self._system_message = system_message
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._context_history = [
            {
                "role": "system",
                "content": self._system_message,
            }
        ]
    
    # Property to access tools from the tool handler
    @property
    def tools(self):
        return self._tool_handler.tools
        
    async def define_system_message(self) -> str:
        return self._system_message
        
    async def model_generate(self, 
                       messages: List[Dict[str, str]],
                       tools: Optional[List[Dict[str, Any]]] = None,
                       response_schema: Optional[Any] = NOT_GIVEN) -> Union[OpenAgentResponse, BaseModel]:
        if tools is None:
            tools = self.tools
            
        #logger.info(f"Tools: {tools}")

        if response_schema:
            response = (
                await self._client.beta.chat.completions.parse(
                    model=self._model,
                    messages=messages,
                    tools=tools,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    top_p=self._top_p,
                    response_format=response_schema,
                )
            ).choices[0].message
            
            if (response.refusal):
                return OpenAgentResponse(**response.model_dump())
            else:
                response = response.parsed
        else:
            response = (
                await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=tools,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    top_p=self._top_p,
                )
            ).choices[0].message
            
            if (response.refusal):
                return OpenAgentResponse(**response.model_dump())
            
        # Extract tool_calls arguments using the tool handler
        tool_calls = self._tool_handler.parse_tool_args(response)
            
        if response_schema:
            return response
        
        response = OpenAgentResponse(**response.model_dump())
        response.tool_calls = tool_calls
        
        return response
        
    async def add_context(self, content: dict):
        self._context_history.append(content)
        return self._context_history
        
    async def extend_context(self, content: List[dict[str, str]]):
        self._context_history.extend(content)
        return self._context_history
    
    # Delegate tool call handling to the tool handler
    async def _handle_tool_call(self, tool_name, **tool_args):
        return self._tool_handler._handle_tool_call(tool_name, **tool_args)