from openagentkit.core.interfaces import AsyncBaseExecutor
from typing import Optional, List, Dict, Any, AsyncGenerator
from openagentkit.core.models.responses import OpenAgentResponse, OpenAgentStreamingResponse
from openagentkit.modules.openai.openai_realtime_service import OpenAIRealtimeService
import websockets
from pydantic import BaseModel
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openagentkit.core.handlers.tool_handler import ToolHandler
from openai._types import NOT_GIVEN
from typing import Callable, Literal
from loguru import logger
import os

class OpenAIRealtimeExecutor(AsyncBaseExecutor):
    def __init__(self, 
                 client: AsyncOpenAI,
                 model: str = "gpt-4o-mini-realtime-preview",
                 voice: Literal["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"] = None,
                 system_message: Optional[str] = None,
                 tools: Optional[List[Callable[..., Any]]] = NOT_GIVEN,
                 api_key: Optional[str] = os.getenv("OPENAI_API_KEY"),
                 temperature: Optional[float] = 0.3,
                 max_tokens: Optional[int] = None,
                 top_p: Optional[float] = None):
        self._llm_service = OpenAIRealtimeService(
            client=client,
            model=model,
            voice=voice,
            system_message=system_message,
            tools=tools,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        self._tools = tools
        self._tool_handler = ToolHandler(tools=tools, type="OpenAIRealtime")

    def clone(self) -> 'OpenAIRealtimeExecutor':
        return OpenAIRealtimeExecutor(
            client=self._llm_service._client,
            model=self._llm_service._model,
            voice=self._llm_service._voice,
            system_message=self._llm_service._system_message,
            tools=self._tools,
            api_key=self._llm_service._api_key,
            temperature=self._llm_service._temperature,
            max_tokens=self._llm_service._max_tokens,
            top_p=self._llm_service._top_p
        )
    
    @property
    def model(self) -> str:
        return self._llm_service._model
    
    @property
    def temperature(self) -> float:
        return self._llm_service.temperature

    @property
    def tools(self) -> List[Dict[str, Any]]:
        return self._llm_service._tool_handler.tools

    async def define_system_message(self, system_message: Optional[str]) -> str:
        return system_message
    
    async def start_up(self):
        await self._llm_service.start_up()

    async def execute(self, 
                      messages: List[Dict[str, str]], 
                      tools: Optional[List[Dict[str, Any]]], 
                      temperature: Optional[float] = None, 
                      max_tokens: Optional[int] = None, 
                      top_p: Optional[float] = None
                      ) -> AsyncGenerator[OpenAgentResponse, None]:
        pass

    async def stream_execute(self, 
                             messages: List[Dict[str, str]], 
                             tools: Optional[List[Dict[str, Any]]], 
                             temperature: Optional[float] = None, 
                             max_tokens: Optional[int] = None, 
                             top_p: Optional[float] = None,
                             **kwargs) -> AsyncGenerator[OpenAgentStreamingResponse, None]:
        temperature = kwargs.get("temperature", temperature)
        if temperature is None:
            temperature = self.temperature
        
        debug = kwargs.get("debug", False)
        
        if tools == NOT_GIVEN:
            tools = self._llm_service._tool_handler.tools  

        stop = False

        latest_message = [messages[-1]] # NOTE: This is a hack for getting ONLY 1 message

        while not stop:
            print(f"Latest Message: {latest_message}")
            response = self._llm_service.model_stream(
                messages=latest_message,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )

            async for chunk in response:
                if chunk.finish_reason == "tool_calls":
                    yield OpenAgentStreamingResponse(
                        role=chunk.role,
                        content=str(chunk.content) if not isinstance(chunk.content, (BaseModel, type(None))) else chunk.content,
                        tool_calls=chunk.tool_calls,
                        usage=chunk.usage,
                    )

                    logger.debug(f"Context: {messages}") if debug else None

                    # Handle the tool call request and get the final response with tool results
                    tool_response = await self._tool_handler.async_handle_tool_request(
                        response=chunk,
                    )

                    latest_message = [tool_response.tool_messages[-1].model_dump()] # NOTE: This is a hack for getting ONLY 1 tool message

                    yield OpenAgentStreamingResponse(
                        role="tool",
                        tool_results=tool_response.tool_messages,
                    )

                elif chunk.finish_reason == "stop":
                    logger.debug(f"Final Chunk: {chunk}") if debug else None
                    if chunk.content:
                        context = self._llm_service.add_context(
                            {
                                "role": "assistant",
                                "content": str(chunk.content),
                            }
                        )
                        logger.debug(f"Context: {context}") if debug else None
                        yield chunk
                        stop = True
                else:
                    yield chunk

    def get_history(self) -> List[Dict[str, Any]]:
        return []

