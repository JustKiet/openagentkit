from typing import Any, Callable, Dict, List, Optional, Union, Generator
import os
from loguru import logger
from openai._types import NOT_GIVEN
from openai import OpenAI
from openagentkit.interfaces.base_executor import BaseExecutor
from openagentkit.modules.openai import OpenAILLMService
from openagentkit.models.responses import OpenAgentResponse, OpenAIStreamingResponse
from openagentkit.handlers.tool_handler import ToolHandler
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
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._tool_handler = ToolHandler(tools=tools)

    def define_system_message(self, message: Optional[str] = None) -> str:
        system_message = message if message is not None else """
            System Message: You are an helpful assistant, try to assist the user in everything.\n
            """
        system_message += f"""
        Current date and time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n
        
        """
        return system_message

    def execute(self, 
                messages: List[Dict[str, str]],
                tools: Optional[List[Dict[str, Any]]] = NOT_GIVEN,
                response_schema: Optional[BaseModel] = NOT_GIVEN,
                verbose: bool = False,
                **kwargs,
               ) -> Union[OpenAgentResponse, BaseModel]:
        kwargs.get("temperature", self._temperature)
        kwargs.get("max_tokens", self._max_tokens)
        kwargs.get("top_p", self._top_p)
        
        if tools == NOT_GIVEN:
            tools = self._llm_service.tools
        
        context = self._llm_service.extend_context(messages)
        
        logger.debug(f"Context: {context}")
        
        # Take user initial request along with the chat history -> response
        response = self._llm_service.model_generate(
            messages=context, 
            tools=tools, 
            response_schema=response_schema
        )
        
        if response.content is not None:
            # Add the response to the context (chat history)
            context = self._llm_service.add_context(
                {
                    "role": response.role,
                    "content": str(response.content),
                }
            )
        
        logger.info(f"Response Received: {response}")

        tool_results = []
        
        if response.tool_calls:
            # Add the tool call request to the context
            context = self._llm_service.add_context(
                {
                    "role": response.role,
                    "tool_calls": response.tool_calls,
                    "content": str(response.content),
                }
            )
            # Handle tool requests and get the final response with tool results
            tool_response = self._tool_handler.handle_tool_request(
                response=response,
            )

            logger.debug(f"Tool Messages in Execute: {tool_response.tool_messages}")

            context = self._llm_service.extend_context(tool_response.tool_messages)

            logger.debug(f"Context: {context}")

            # Generate the final response with the tool results
            response = self._llm_service.model_generate(
                messages=context,
                tools=tools, 
                response_schema=response_schema
            )

        # Add the final response to the context (chat history)
        self._llm_service.add_context(
            {
                "role": response.role,
                "content": str(response.content),
            }
        )
        
        logger.debug(f"Final Response: {response}")
        
        # If there is no response, return an error
        if not response:
            logger.error("No response from the model")
            return OpenAgentResponse(
                role="assistant",
                content="",
                tool_results=tool_results,
                refusal="No response from the model",
                audio=None,
            )
        
        return OpenAgentResponse(
            role=response.role,
            content=str(response.content),
            tool_calls=response.tool_calls,
            tool_results=tool_results,
            refusal=response.refusal,
            audio=response.audio,
            usage=response.usage,
        )

    def stream_execute(self, 
                      messages: List[Dict[str, str]],
                      tools: Optional[List[Dict[str, Any]]] = NOT_GIVEN,
                      response_schema: Optional[BaseModel] = NOT_GIVEN,
                      **kwargs,
                      ) -> Generator[OpenAIStreamingResponse, None, None]:
        kwargs.get("temperature", self._temperature)
        kwargs.get("max_tokens", self._max_tokens)
        kwargs.get("top_p", self._top_p)
        
        if tools == NOT_GIVEN:
            tools = self._llm_service.tools

        context = self._llm_service.extend_context(messages)

        logger.debug(f"Context: {context}")

        response_generator = self._llm_service.model_stream(
            messages=context,
            tools=tools,
            response_schema=response_schema,
        )

        final_response_generator = None
        
        for chunk in response_generator:
            if chunk.finish_reason == "tool_calls":
                # Add the llm tool call request to the context
                context = self._llm_service.add_context(
                    {
                        "role": "assistant",
                        "tool_calls": chunk.tool_calls,
                        "content": str(chunk.content),
                    }
                )

                logger.debug(f"Context: {context}")

                notification = self._tool_handler.handle_notification(chunk)

                if notification:
                    yield notification

                # Handle the tool call request and get the final response with tool results
                tool_response = self._tool_handler.handle_tool_request(
                    response=chunk,
                )
                
                logger.debug(f"Tool Messages in Execute: {tool_response.tool_messages}")

                context = self._llm_service.extend_context(tool_response.tool_messages)
                
                logger.debug(f"Context in Stream Execute: {context}")
                
                final_response_generator = self._llm_service.model_stream(
                    messages=context,
                    tools=tools,
                    response_schema=response_schema,
                )

                for chunk in final_response_generator:
                    if chunk.content:
                        context = self._llm_service.add_context(
                            {
                                "role": "assistant", 
                                "content": chunk.content
                            }
                        )
                        logger.info(f"Context: {context}")
                    yield chunk

            elif chunk.finish_reason == "stop":
                if chunk.content:
                    context = self._llm_service.add_context(
                        {
                            "role": "assistant", 
                            "content": chunk.content
                        }
                    )
                    logger.info(f"Context: {context}")
                    yield chunk
            else:
                yield chunk