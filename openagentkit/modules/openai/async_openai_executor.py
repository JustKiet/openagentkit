from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union
import os
from loguru import logger
from openai._types import NOT_GIVEN
from openai import AsyncOpenAI
from openagentkit.interfaces.async_base_executor import AsyncBaseExecutor
from openagentkit.modules.openai.async_openai_llm_service import AsyncOpenAILLMService
from openagentkit.models.responses import OpenAgentResponse, OpenAIStreamingResponse
from openagentkit.handlers.tool_handler import ToolHandler
from pydantic import BaseModel
import json
import datetime

class AsyncOpenAIExecutor(AsyncBaseExecutor):
    """
    An asynchronous executor (agentic) module for OpenAI models.

    Args:
        client (AsyncOpenAI): The AsyncOpenAI client.
        model (str): The model to use. (default: "gpt-4o-mini")
        system_message (Optional[str]): The system message to use. (default: None)
        tools (Optional[List[Callable[..., Any]]]): The tools to use. (default: NOT_GIVEN)
        api_key (Optional[str]): The API key to use. (default: os.getenv("OPENAI_API_KEY"))
        temperature (Optional[float]): The temperature to use. (default: 0.3)
        max_tokens (Optional[int]): The maximum number of tokens to use. (default: None)
        top_p (Optional[float]): The top p to use. (default: None)

    Example:
    ```python
    from openagentkit.modules.openai import AsyncOpenAIExecutor
    from openagentkit.tools import duckduckgo_search_tool
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    executor = AsyncOpenAIExecutor(client=client, tools=[duckduckgo_search_tool])
    response = await executor.execute(messages=[{"role": "user", "content": "What is Quantum Mechanics?"}])
    ```

    """
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
        self._tool_handler = ToolHandler(tools=tools)

    def get_history(self) -> List[Dict[str, Any]]:
        return self._llm_service.history

    def define_system_message(self, message: Optional[str] = None) -> str:
        """
        Define the system message for the OpenAI model.

        Args:
            message (Optional[str]): The system message to use. (default: None)

        Returns:
            str: The system message.
        """
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
                    ) -> OpenAgentResponse:
        """
        Asynchronously execute the OpenAI model and return an OpenAgentResponse object.

        Args:
            messages (List[Dict[str, str]]): The messages to send to the model.
            tools (Optional[List[Dict[str, Any]]]): The tools to use in the response.
            response_schema (Optional[BaseModel]): The schema to use in the response.
            verbose (bool): Whether to print the response. (default: False)

        Returns:
            An OpenAgentResponse object.
        """
        kwargs.get("temperature", self._temperature)
        kwargs.get("max_tokens", self._max_tokens)
        kwargs.get("top_p", self._top_p)
        
        if tools == NOT_GIVEN:
            tools = self._llm_service.tools
        
        context = await self._llm_service.extend_context(messages)
        
        logger.debug(f"Context: {context}")
        
        stop = False
        
        while not stop:
            # Take user intial request along with the chat history -> response
            response = await self._llm_service.model_generate(
                messages=context, 
                tools=tools, 
                response_schema=response_schema
            )

            logger.info(f"Response Received: {response}")

            if response.content is not None:
                # Add the response to the context (chat history)
                context = await self._llm_service.add_context(
                    {
                        "role": response.role,
                        "content": str(response.content),
                    }
                )

            tool_results = []
            
            if response.tool_calls:
                # Add the tool call request to the context
                context = await self._llm_service.add_context(
                    {
                        "role": response.role,
                        "tool_calls": response.tool_calls,
                        "content": str(response.content),
                    }
                )
                # Handle tool requests abd get the final response with tool results
                tool_response = self._tool_handler.handle_tool_request(
                    response=response,
                )

                logger.debug(f"Tool Messages in Execute: {tool_response.tool_messages}")

                context = await self._llm_service.extend_context(tool_response.tool_messages)

                logger.debug(f"Context: {context}")
            
            else:
                stop = True

        # Add the final response to the context (chat history)
        await self._llm_service.add_context(
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

    async def stream_execute(self, 
                             messages: List[Dict[str, str]],
                             tools: Optional[List[Dict[str, Any]]] = NOT_GIVEN,
                             response_schema: Optional[BaseModel] = NOT_GIVEN,
                             **kwargs,
                             ) -> AsyncGenerator[OpenAIStreamingResponse, None]:
        kwargs.get("temperature", self._temperature)
        kwargs.get("max_tokens", self._max_tokens)
        kwargs.get("top_p", self._top_p)
        
        if tools == NOT_GIVEN:
            tools = self._llm_service.tools

        stop = False

        context = await self._llm_service.extend_context(messages)

        while not stop: 

            logger.debug(f"Context: {context}")

            response_generator = self._llm_service.model_stream(
                messages=context,
                tools=tools,
                response_schema=response_schema,
            )
            
            async for chunk in response_generator:
                if chunk.finish_reason == "tool_calls":
                    # Add the llm tool call request to the context
                    context = await self._llm_service.add_context(
                        {
                            "role": "assistant",
                            "tool_calls": chunk.tool_calls,
                            "content": str(chunk.content),
                        }
                    )

                    logger.debug(f"Context: {context}")

                    # Handle the notification (if any) from the tool call chunk
                    notification = self._tool_handler.handle_notification(chunk)

                    # If there is a tool call notification but NO CONTENT, yield the notification
                    if notification and not chunk.content:
                        yield notification

                    # Handle the tool call request and get the final response with tool results
                    tool_response = self._tool_handler.handle_tool_request(
                        response=chunk,
                    )

                    logger.debug(f"Tool Messages in Execute: {tool_response.tool_messages}")

                    context = await self._llm_service.extend_context(tool_response.tool_messages)
                    
                    logger.debug(f"Context in Stream Execute: {context}")

                elif chunk.finish_reason == "stop":
                    if chunk.content:
                        context = await self._llm_service.add_context(
                            {
                                "role": "assistant",
                                "content": str(chunk.content),
                            }
                        )
                        logger.info(f"Context: {context}")
                        yield chunk
                        stop = True
                else:
                    yield chunk
