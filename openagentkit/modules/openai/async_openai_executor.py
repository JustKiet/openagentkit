from typing import Any, Callable, Dict, List, Optional, Union
import os
from loguru import logger
from openai._types import NOT_GIVEN
from openai import AsyncOpenAI
from openagentkit.interfaces.async_base_executor import AsyncBaseExecutor
from openagentkit.modules.openai.async_openai_llm_service import AsyncOpenAILLMService
from openagentkit.models import OpenAgentResponse
from pydantic import BaseModel
import json
import datetime

class AsyncOpenAIExecutor(AsyncBaseExecutor):
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
        self._llm_service = AsyncOpenAILLMService(
            client=client,
            model=model,
            system_message=self.define_system_message(system_message),
            tools=tools,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p

    def define_system_message(self, message: Optional[str] = None) -> str:
        system_message = message if message is not None else """
            System Message: You are an helpful assistant, try to assist the user in everything.\n
            """
        system_message += f"""
        Current date and time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n
        
        """
        return system_message

    async def execute(self, 
                messages: List[Dict[str, str]],
                tools: Optional[List[Dict[str, Any]]] = NOT_GIVEN,
                response_schema: Optional[BaseModel] = NOT_GIVEN,
                **kwargs,
               ) -> Union[OpenAgentResponse, BaseModel]:
        kwargs.get("temperature", self._temperature)
        kwargs.get("max_tokens", self._max_tokens)
        kwargs.get("top_p", self._top_p)
        
        if tools == NOT_GIVEN:
            tools = self._llm_service.tools
        
        context = await self._llm_service.extend_context(messages)
        
        logger.debug(f"Context: {context}")
        
        response = (await self._llm_service.model_generate(
            messages=context, 
            tools=tools, 
            response_schema=response_schema
        )).model_dump()
        
        response = {
            "role": "assistant",
            "content": response,
        } if response_schema else response
        
        context = await self._llm_service.add_context(response)
        
        logger.info(f"Response Received: {response}" )
        
        if response.get("tool_calls", None) != None and type(response) == dict:
            for tool_call in response.get("tool_calls"):
                tool_call_id = tool_call.get("id")
                tool_name = tool_call.get("function").get("name")
                tool_args = eval(tool_call.get("function").get("arguments"))
                tool_result = (await self._llm_service._handle_tool_call(tool_name, **tool_args)).model_dump()
                
                logger.info(f"Tool Result: {tool_result}")
                
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(tool_result),
                }
                
                context = await self._llm_service.add_context(tool_message)
                
            response_with_tool_context = (await self._llm_service.model_generate(
                messages=context, 
                tools=tools, 
                response_schema=response_schema
            )).model_dump()
            
            await self._llm_service.add_context(response_with_tool_context)
            
        final_response = self._llm_service._context_history[-1]
        
        logger.debug(f"Final Response: {final_response}")
        
        if response_schema:
            return response_schema(**final_response.get("content"))
        
        if response:
            return OpenAgentResponse(**final_response)