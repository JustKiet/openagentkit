from typing import Any, Callable, Dict, List, Optional, Union
import os
from loguru import logger
from openai._types import NOT_GIVEN
from openai import OpenAI
from openagentkit.interfaces.base_executor import BaseExecutor
from openagentkit.modules.openai import OpenAILLMService
from openagentkit.models import OpenAgentResponse
from pydantic import BaseModel
import json
import datetime

class OpenAIExecutor(BaseExecutor):
    def __init__(self,
                 client: OpenAI,
                 model: str = "gpt-4o-mini",
                 system_message: Optional[str] = None,
                 tools: Optional[List[Callable[..., Any]]] = NOT_GIVEN,
                 api_key: Optional[str] = os.getenv("OPENAI_API_KEY"),
                 temperature: Optional[float] = 0.3,
                 max_tokens: Optional[int] = None,
                 top_p: Optional[float] = None,
                 *args,
                 **kwargs):
        # Create an instance of OpenAILLMService instead of inheriting from it
        self._llm_service = OpenAILLMService(
            client=client,
            model=model,
            system_message=self.define_system_message(system_message),
            tools=tools,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        # Store configuration parameters
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._context_history = []

    def define_system_message(self, message: Optional[str] = None) -> str:
        if message is not None:
            self._system_message = message
        else:
            self._system_message = """
            System Message: You are an helpful assistant, try to assist the user in everything.\n
            """
        self._system_message += f"""
        Current date and time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n
        
        """
        return self._system_message

    def extend_context(self, messages):
        # Delegate to the LLM service
        return self._llm_service.extend_context(messages)
    
    def add_context(self, message):
        # Delegate to the LLM service
        return self._llm_service.add_context(message)
    
    def _handle_tool_call(self, tool_name, **tool_args):
        # Delegate to the LLM service
        return self._llm_service._handle_tool_call(tool_name, **tool_args)

    def execute(self, 
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
        
        context = self.extend_context(messages)
        
        logger.debug(f"Context: {context}")
        
        response = self._llm_service.model_generate(
            messages=context, 
            tools=tools, 
            response_schema=response_schema
        ).model_dump()
        
        response = {
            "role": "assistant",
            "content": response,
        } if response_schema else response
        
        context = self.add_context(response)
        
        logger.info(f"Response Received: {response}" )
        
        if response.get("tool_calls", None) != None and type(response) == dict:
            for tool_call in response.get("tool_calls"):
                tool_call_id = tool_call.get("id")
                tool_name = tool_call.get("function").get("name")
                tool_args = eval(tool_call.get("function").get("arguments"))
                tool_result = self._handle_tool_call(tool_name, **tool_args).model_dump()
                
                logger.info(f"Tool Result: {tool_result}")
                
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(tool_result),
                }
                
                context = self.add_context(tool_message)
                
            response_with_tool_context = self._llm_service.model_generate(
                messages=context, 
                tools=tools, 
                response_schema=response_schema
            ).model_dump()
            
            self.add_context(response_with_tool_context)
            
        final_response = self._llm_service._context_history[-1]
        
        logger.debug(f"Final Response: {final_response}")
        
        if response_schema:
            return response_schema(**final_response.get("content"))
        
        if response:
            return OpenAgentResponse(**final_response)