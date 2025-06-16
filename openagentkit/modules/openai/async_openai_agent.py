from typing import Any, AsyncGenerator, Dict, List, Optional
import os
from loguru import logger
from openai._types import NOT_GIVEN
from openai import AsyncOpenAI
from openagentkit.core.interfaces.async_base_agent import AsyncBaseAgent
from openagentkit.modules.openai.async_openai_llm_service import AsyncOpenAILLMService
from openagentkit.core.models.responses import OpenAgentResponse, OpenAgentStreamingResponse
from openagentkit.core.tools.tool_handler import ToolHandler
from openagentkit.core.tools.base_tool import Tool
from openagentkit.modules.openai import OpenAIAudioFormats, OpenAIAudioVoices
from pydantic import BaseModel
from mcp import ClientSession

class AsyncOpenAIAgent(AsyncBaseAgent):
    """
    An asynchronous Agent (agentic) module for OpenAI models.

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
    from openagentkit.modules.openai import AsyncOpenAIAgent
    from openagentkit.tools import duckduckgo_search_tool
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    agent = AsyncOpenAIAgent(client=client, tools=[duckduckgo_search_tool])
    response = await agent.execute(messages=[{"role": "user", "content": "What is Quantum Mechanics?"}])
    ```

    """
    def __init__(
        self,
        client: Optional[AsyncOpenAI] = None,
        model: str = "gpt-4o-mini",
        system_message: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        api_key: Optional[str] = os.getenv("OPENAI_API_KEY"),
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        context_history = kwargs.get("context_history", None)
        super().__init__(system_message=system_message, context_history=context_history)
        self._llm_service = AsyncOpenAILLMService(
            client=client,
            model=model,
            tools=tools,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        self._tools = tools
        self._tool_handler = ToolHandler(
            tools=tools
        )

    @property
    def model(self) -> str:
        return self._llm_service.model

    @property
    def temperature(self) -> float:
        return self._llm_service.temperature

    @property
    def max_tokens(self) -> int | None:
        return self._llm_service.max_tokens
    
    @property
    def top_p(self) -> float | None:
        return self._llm_service.top_p
    
    @property
    def tools(self) -> List[Dict[str, Any]] | None:
        return self._llm_service.tools

    async def connect_to_mcp(self, mcp_sessions: list[ClientSession]) -> None:
        self._tool_handler = await ToolHandler.from_mcp(sessions=mcp_sessions, additional_tools=self._tools)
        self._llm_service.tool_handler = self._tool_handler
    
    def clone(self) -> 'AsyncOpenAIAgent':
        """
        Clone the AsyncOpenAIAgent object.

        Returns:
            A new AsyncOpenAIAgent object with the same parameters.
        """
        return AsyncOpenAIAgent(
            client=self._llm_service.client,
            model=self._llm_service.model,
            system_message=self._system_message,
            tools=self._tools,
            api_key=self._llm_service.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )

    async def execute(
        self, 
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        response_schema: Optional[type[BaseModel]] = None,
        audio: Optional[bool] = False,
        audio_format: Optional[OpenAIAudioFormats] = "pcm16",
        audio_voice: Optional[OpenAIAudioVoices] = "alloy",
        **kwargs: Any,
    ) -> AsyncGenerator[OpenAgentResponse, None]:
        """
        Asynchronously execute the OpenAI model and return an OpenAgentResponse object.

        Args:
            messages (List[Dict[str, str]]): The messages to send to the model.
            tools (Optional[List[Dict[str, Any]]]): The tools to use in the response.
            response_schema (Optional[type[BaseModel]]): The schema to use in the response.
            temperature (Optional[float]): The temperature to use in the response.
            max_tokens (Optional[int]): The maximum number of tokens to use in the response.
            top_p (Optional[float]): The top p to use in the response.
            audio (Optional[bool]): Whether to use audio in the response.
            audio_format (Optional[Literal['wav', 'mp3', 'flac', 'opus', 'pcm16']]): The format to use in the response.
            audio_voice (Optional[Literal["alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"]]): The voice to use in the response.

        Returns:
            An OpenAgentResponse asynchronous generator.

        Example:
        ```python
        from openagentkit.modules.openai import AsyncOpenAIAgent
        from openagentkit.tools import duckduckgo_search_tool
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        Agent = AsyncOpenAIAgent(client=client, tools=[duckduckgo_search_tool])
        async for response in Agent.execute(
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
            tools = self._llm_service.tool_handler.tools
        
        context: list[dict[str, Any]] = await self.extend_context(messages)
        
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
                audio=audio,
                audio_format=audio_format,
                audio_voice=audio_voice,
            )

            logger.info(f"Response Received: {response}") if debug else None

            if response.content is not None:
                # Add the response to the context (chat history)
                context = await self.add_context(
                    {
                        "role": response.role,
                        "content": str(response.content),
                    }
                )

            tool_results: list[Any] = []
            
            if response.tool_calls:
                tool_calls: list[dict[str, str]] = [tool_call.model_dump() for tool_call in response.tool_calls]
                # Add the tool call request to the context
                context = await self.add_context(
                    {
                        "role": response.role,
                        "tool_calls": tool_calls,
                        "content": str(response.content),
                    }
                )

                yield OpenAgentResponse(
                    role=response.role,
                    content=str(response.content) if not isinstance(response.content, (BaseModel, type(None))) else response.content,
                    tool_calls=response.tool_calls,
                    refusal=response.refusal,
                    usage=response.usage,
                )

                # Handle tool requests abd get the final response with tool results
                tool_response = await self._tool_handler.async_handle_tool_request(
                    tool_calls=response.tool_calls,
                )

                yield OpenAgentResponse(
                    role="tool",
                    tool_results=tool_response.tool_results,
                )

                logger.debug(f"Tool Messages in Execute: {tool_response.tool_messages}") if debug else None

                context = await self.extend_context([tool_message.model_dump() for tool_message in tool_response.tool_messages] if tool_response.tool_messages else [])

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

    async def stream_execute(
        self, 
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        response_schema: Optional[type[BaseModel]] = None,
        audio: Optional[bool] = False,
        audio_format: Optional[OpenAIAudioFormats] = "pcm16",
        audio_voice: Optional[OpenAIAudioVoices] = "alloy",
        **kwargs: Any,
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
            audio_format (Optional[Literal['wav', 'mp3', 'flac', 'opus', 'pcm16']]): The format to use in the response.
            audio_voice (Optional[Literal["alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"]]): The voice to use in the response.

        Returns:
            An OpenAgentStreamingResponse asynchronous generator.

        Example:
        ```python
        from openagentkit.modules.openai import AsyncOpenAIAgent
        from openagentkit.tools import duckduckgo_search_tool
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        Agent = AsyncOpenAIAgent(client=client, tools=[duckduckgo_search_tool])

        async for chunk in Agent.stream_execute(
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

        context: list[dict[str, Any]] = await self.extend_context(messages)

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
                if chunk.finish_reason == "tool_calls" and chunk.tool_calls:
                    tool_calls: list[dict[str, Any]] = [tool_call.model_dump() for tool_call in chunk.tool_calls] if chunk.tool_calls else []
                    # Add the llm tool call request to the context
                    context = await self.add_context(
                        {
                            "role": "assistant",
                            "tool_calls": tool_calls,
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

                    # Handle the tool call request and get the final response with tool results
                    tool_response = await self._tool_handler.async_handle_tool_request(
                        tool_calls=chunk.tool_calls,
                    )

                    yield OpenAgentStreamingResponse(
                        role="tool",
                        tool_results=tool_response.tool_results,
                    )

                    logger.debug(f"Tool Messages in Execute: {tool_response.tool_messages}") if debug else None

                    context = await self.extend_context([tool_message.model_dump() for tool_message in tool_response.tool_messages] if tool_response.tool_messages else [])
                    
                    logger.debug(f"Context in Stream Execute: {context}") if debug else None

                elif chunk.finish_reason == "stop":
                    logger.debug(f"Final Chunk: {chunk}") if debug else None
                    if chunk.content:
                        context = await self.add_context(
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
