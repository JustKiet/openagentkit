from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union
import os
from loguru import logger
from openai._types import NOT_GIVEN
from openai import AsyncOpenAI
from openagentkit.core.interfaces.async_base_executor import AsyncBaseExecutor
from openagentkit.modules.openai.async_openai_llm_service import AsyncOpenAILLMService
from openagentkit.core.models.responses import OpenAgentResponse, OpenAgentStreamingResponse
from openagentkit.core.handlers.tool_handler import ToolHandler
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
        self._tool_handler = ToolHandler(tools=tools)

    @property
    def model(self) -> str:
        return self._llm_service.model

    @property
    def temperature(self) -> float:
        return self._llm_service.temperature

    @property
    def max_tokens(self) -> int:
        return self._llm_service.max_tokens
    
    @property
    def top_p(self) -> float:
        return self._llm_service.top_p
    
    def clone(self) -> 'AsyncOpenAIExecutor':
        """
        Clone the AsyncOpenAIExecutor object.

        Returns:
            A new AsyncOpenAIExecutor object with the same parameters.
        """
        return AsyncOpenAIExecutor(
            client=self._llm_service.client,
            model=self._llm_service.model,
            system_message=self._llm_service.system_message,
            tools=self._llm_service.tools,
            api_key=self._llm_service.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )

    def get_history(self) -> List[Dict[str, Any]]:
        return self._llm_service.history
    
    def clear_history(self) -> list[Dict[str, Any]]:
        """
        Clear the chat history leaving only the system message.
        """
        return self._llm_service.clear_context()

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
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      top_p: Optional[float] = None,
                      **kwargs,
                    ) -> AsyncGenerator[OpenAgentResponse, None]:
        """
        Asynchronously execute the OpenAI model and return an OpenAgentResponse object.

        Args:
            messages (List[Dict[str, str]]): The messages to send to the model.
            tools (Optional[List[Dict[str, Any]]]): The tools to use in the response.
            response_schema (Optional[BaseModel]): The schema to use in the response.
            temperature (Optional[float]): The temperature to use in the response.
            max_tokens (Optional[int]): The maximum number of tokens to use in the response.
            top_p (Optional[float]): The top p to use in the response.

        Returns:
            An OpenAgentResponse asynchronous generator.

        Example:
        ```python
        from openagentkit.modules.openai import AsyncOpenAIExecutor
        from openagentkit.tools import duckduckgo_search_tool
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        executor = AsyncOpenAIExecutor(client=client, tools=[duckduckgo_search_tool])
        async for response in executor.execute(
            messages=[{"role": "user", "content": "What is Quantum Mechanics?"}]
        ):
            print(response)
        ```
        """
        temperature = kwargs.get("temperature", temperature)
        if temperature is None:
            temperature = self.temperature

        max_tokens = kwargs.get("max_tokens", max_tokens)
        if max_tokens is None:
            max_tokens = self.max_tokens

        top_p = kwargs.get("top_p", top_p)
        if top_p is None:
            top_p = self.top_p
        
        debug = kwargs.get("debug", False)
        
        if tools == NOT_GIVEN:
            tools = self._llm_service.tools
        
        context = self._llm_service.extend_context(messages)
        
        logger.debug(f"Context: {context}") if debug else None
        
        stop = False
        
        while not stop:
            # Take user intial request along with the chat history -> response
            response = await self._llm_service.model_generate(
                messages=context, 
                tools=tools, 
                response_schema=response_schema,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )

            logger.info(f"Response Received: {response}") if debug else None

            if response.content is not None:
                # Add the response to the context (chat history)
                context = self._llm_service.add_context(
                    {
                        "role": response.role,
                        "content": str(response.content),
                    }
                )

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

                print(type(response.content))

                yield OpenAgentResponse(
                    role=response.role,
                    content=str(response.content) if not isinstance(response.content, (BaseModel, type(None))) else response.content,
                    tool_calls=response.tool_calls,
                    refusal=response.refusal,
                    usage=response.usage,
                )

                # Handle tool requests abd get the final response with tool results
                tool_response = self._tool_handler.handle_tool_request(
                    response=response,
                )

                yield OpenAgentResponse(
                    role="tool",
                    tool_results=tool_response.tool_results,
                )

                logger.debug(f"Tool Messages in Execute: {tool_response.tool_messages}") if debug else None

                context = self._llm_service.extend_context([tool_message.model_dump() for tool_message in tool_response.tool_messages])

                logger.debug(f"Context: {context}") if debug else None
            
            else:
                stop = True
            
            if response.content is not None:        
                # If there is no response, return an error
                if not response:
                    logger.error("No response from the model")
                    yield OpenAgentResponse(
                        role="assistant",
                        content="",
                        tool_results=tool_results,
                        refusal="No response from the model",
                        audio=None,
                    )

                yield OpenAgentResponse(
                    role=response.role,
                    content=str(response.content) if not isinstance(response.content, (BaseModel, type(None))) else response.content,
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
                             temperature: Optional[float] = None,
                             max_tokens: Optional[int] = None,
                             top_p: Optional[float] = None,
                             audio: Optional[bool] = False,
                             audio_format: Optional[str] = "pcm16",
                             audio_voice: Optional[str] = "alloy",
                             **kwargs,
                             ) -> AsyncGenerator[OpenAgentStreamingResponse, None]:
        """
        Asynchronously stream the OpenAI model and return an OpenAgentStreamingResponse object.

        Args:
            messages (List[Dict[str, str]]): The messages to send to the model.
            tools (Optional[List[Dict[str, Any]]]): The tools to use in the response.
            response_schema (Optional[BaseModel]): The schema to use in the response.
            temperature (Optional[float]): The temperature to use in the response.
            max_tokens (Optional[int]): The maximum number of tokens to use in the response.
            top_p (Optional[float]): The top p to use in the response.
            audio (Optional[bool]): Whether to use audio in the response.
            audio_format (Optional[str]): The format to use in the response.
            audio_voice (Optional[str]): The voice to use in the response.

        Returns:
            An OpenAgentStreamingResponse asynchronous generator.

        Example:
        ```python
        from openagentkit.modules.openai import AsyncOpenAIExecutor
        from openagentkit.tools import duckduckgo_search_tool
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        executor = AsyncOpenAIExecutor(client=client, tools=[duckduckgo_search_tool])

        async for chunk in executor.stream_execute(
            messages=[{"role": "user", "content": "What is Quantum Mechanics?"}]
        ):
            print(chunk)
        ```
        """
        temperature = kwargs.get("temperature", temperature)
        if temperature is None:
            temperature = self.temperature

        max_tokens = kwargs.get("max_tokens", max_tokens)
        if max_tokens is None:
            max_tokens = self.max_tokens

        top_p = kwargs.get("top_p", top_p)
        if top_p is None:
            top_p = self.top_p
            
        debug = kwargs.get("debug", False)
        
        if tools == NOT_GIVEN:
            tools = self._llm_service.tools

        stop = False

        context = self._llm_service.extend_context(messages)

        while not stop:
            logger.debug(f"Context: {context}") if debug else None

            response_generator = self._llm_service.model_stream(
                messages=context,
                tools=tools,
                response_schema=response_schema,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                audio=audio,
                audio_format=audio_format,
                audio_voice=audio_voice,
            )
            
            async for chunk in response_generator:
                if chunk.finish_reason == "tool_calls":
                    # Add the llm tool call request to the context
                    context = self._llm_service.add_context(
                        {
                            "role": "assistant",
                            "tool_calls": chunk.tool_calls,
                            "content": str(chunk.content),
                        }
                    )

                    yield OpenAgentStreamingResponse(
                        role=chunk.role,
                        content=str(chunk.content) if not isinstance(chunk.content, (BaseModel, type(None))) else chunk.content,
                        tool_calls=chunk.tool_calls,
                        usage=chunk.usage,
                    )

                    logger.debug(f"Context: {context}") if debug else None

                    # Handle the notification (if any) from the tool call chunk
                    notification = self._tool_handler.handle_notification(chunk)

                    # If there is a tool call notification but NO CONTENT, yield the notification
                    if notification and not chunk.content:
                        yield notification

                    # Handle the tool call request and get the final response with tool results
                    tool_response = self._tool_handler.handle_tool_request(
                        response=chunk,
                    )

                    yield OpenAgentStreamingResponse(
                        role="tool",
                        tool_results=tool_response.tool_results,
                    )

                    logger.debug(f"Tool Messages in Execute: {tool_response.tool_messages}") if debug else None

                    context = self._llm_service.extend_context([tool_message.model_dump() for tool_message in tool_response.tool_messages])
                    
                    logger.debug(f"Context in Stream Execute: {context}") if debug else None

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
