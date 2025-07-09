from typing import Any, AsyncGenerator, Dict, List, Optional, Literal
import os
from openai import AsyncOpenAI
from openagentkit.core.interfaces import AsyncBaseAgent, BaseContextStore
from openagentkit.core.context import InMemoryContextStore
from openagentkit.modules.openai.async_openai_llm_service import AsyncOpenAILLMService
from openagentkit.core.models.responses import OpenAgentResponse, OpenAgentStreamingResponse
from openagentkit.core.tools.tool_handler import ToolHandler
from openagentkit.core.tools.base_tool import Tool
from openagentkit.modules.openai import OpenAIAudioFormats, OpenAIAudioVoices
from pydantic import BaseModel
from mcp import ClientSession
import logging
import uuid

logger = logging.getLogger(__name__)

class AsyncOpenAIAgent(AsyncBaseAgent):
    def __init__(
        self,
        client: Optional[AsyncOpenAI] = None,
        model: str = "gpt-4o-mini",
        system_message: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        context_store: Optional[BaseContextStore] = None,
        api_key: Optional[str] = os.getenv("OPENAI_API_KEY"),
        temperature: Optional[float] = 1.0,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = 1.0,
        thread_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> None:
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
        
        if not context_store:
            context_store = InMemoryContextStore()
        
        self.context_store: BaseContextStore = context_store

        if not thread_id:
            thread_id = str(uuid.uuid4())

        self._thread_id = thread_id

        if not agent_id:
            agent_id = str(uuid.uuid4())
        
        self._agent_id = agent_id

        self._system_message = system_message or "You are a helpful assistant."

        self.context_store.init_context(
            thread_id=self._thread_id,
            agent_id=self._agent_id,
            system_message=self._system_message,
        )

    @property
    def system_message(self) -> str:
        return self._system_message
    
    @system_message.setter
    def system_message(self, value: str) -> None:
        """
        Set the system message for the agent.

        :param value: The system message to set.
        """
        self._system_message = value
        self.context_store.update_system_message(
            thread_id=self._thread_id,
            agent_id=self._agent_id,
            system_message=value,
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
    
    @property
    def tool_handler(self) -> ToolHandler:
        """
        Get the tool handler for the agent.

        Returns:
            The tool handler.
        """
        return self._tool_handler

    async def connect_to_mcp(self, mcp_sessions: list[ClientSession]) -> None:
        self._tool_handler = await ToolHandler.from_mcp(sessions=mcp_sessions, additional_tools=self._tools)
        self._llm_service.tool_handler = self._tool_handler

    @property
    def thread_id(self) -> str:
        """
        Get the thread ID for the agent.

        Returns:
            The thread ID.
        """
        return self._thread_id
    
    @property
    def agent_id(self) -> str:
        """
        Get the agent ID for the agent.

        Returns:
            The agent ID.
        """
        return self._agent_id

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
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[OpenAgentResponse, None]:
        """
        Asynchronously execute the OpenAI model and return an OpenAgentResponse object.

        :param list[dict[str, str]] messages: The messages to send to the model.
        :param list[dict[str, Any]] tools: The tools to use in the response.
        :param type[BaseModel] response_schema: The schema to use in the response.
        :param float temperature: The temperature to use in the response.
        :param int max_tokens: The maximum number of tokens to use in the response.
        :param float top_p: The top p to use in the response.
        :param bool audio: Whether to use audio in the response.
        :param OpenAIAudioFormats audio_format: The format to use in the response.
        :param OpenAIAudioVoices audio_voice: The voice to use in the response.
        :param Optional[Literal["low", "medium", "high"]] reasoning_effort: The reasoning effort to use in the response (Only for reasoning models).
        :param kwargs: Additional keyword arguments.
        :return: An OpenAgentResponse asynchronous generator.
        :rtype: AsyncGenerator[OpenAgentResponse, None]
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        thread_id = kwargs.get("thread_id", self._thread_id)
        
        if not tools:
            tools = self._llm_service.tools

        if thread_id != self._thread_id:
            self.context_store.init_context(
                thread_id=thread_id,
                agent_id=self._agent_id,
                system_message=self._system_message,
            )

        if self.context_store.get_system_message(thread_id=thread_id) != self._system_message:
            self.context_store.update_system_message(
                thread_id=thread_id,
                agent_id=self._agent_id,
                system_message=self._system_message,
            )
        
        context: list[dict[str, Any]] = self.context_store.extend_context(
            thread_id=thread_id,
            agent_id=self._agent_id,
            content=messages
        ).history
        
        logger.debug(f"Context: {context}")
        
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
                reasoning_effort=reasoning_effort
            )

            logger.debug(f"Response Received: {response}")

            if response.content is not None:
                # Add the response to the context (chat history)
                context = self.context_store.add_context(
                    thread_id=thread_id,
                    agent_id=self._agent_id,
                    content={
                        "role": response.role,
                        "content": str(response.content),
                    }
                ).history

            tool_results: list[Any] = []
            
            if response.tool_calls:
                tool_calls: list[dict[str, str]] = [tool_call.model_dump() for tool_call in response.tool_calls]
                # Add the tool call request to the context
                context = self.context_store.add_context(
                    thread_id=thread_id,
                    agent_id=self._agent_id,
                    content={
                        "role": response.role,
                        "tool_calls": tool_calls,
                        "content": str(response.content),
                    }
                ).history

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

                logger.debug(f"Tool Messages in Execute: {tool_response.tool_messages}")

                context = self.context_store.extend_context(
                    thread_id=thread_id,
                    agent_id=self._agent_id,
                    content=[
                        tool_message.model_dump() 
                        for tool_message in tool_response.tool_messages
                    ] 
                    if tool_response.tool_messages else []
                ).history

                logger.debug(f"Context: {context}")
            
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
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[OpenAgentStreamingResponse, None]:
        """
        Asynchronously stream the OpenAI model and return an OpenAgentStreamingResponse object.

        :param list[dict[str, str]] messages: The messages to send to the model.
        :param list[dict[str, Any]] tools: The tools to use in the response.
        :param type[BaseModel] response_schema: The schema to use in the response.
        :param float temperature: The temperature to use in the response.
        :param int max_tokens: The maximum number of tokens to use in the response.
        :param float top_p: The top p to use in the response.
        :param bool audio: Whether to use audio in the response.
        :param OpenAIAudioFormats audio_format: The format to use in the response.
        :param OpenAIAudioVoices audio_voice: The voice to use in the response.
        :param Optional[Literal["low", "medium", "high"]] reasoning_effort: The reasoning effort to use in the response (Only for reasoning models).
        :param kwargs: Additional keyword arguments.
        :return: An OpenAgentStreamingResponse asynchronous generator.
        :rtype: AsyncGenerator[OpenAgentStreamingResponse, None]
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        thread_id = kwargs.get("thread_id", self._thread_id)
        
        if not tools:
            tools = self._llm_service.tools

        stop = False

        if thread_id != self._thread_id:
            self.context_store.init_context(
                thread_id=thread_id,
                agent_id=self._agent_id,
                system_message=self._system_message,
            )

        context: list[dict[str, Any]] = self.context_store.extend_context(
            thread_id=thread_id,
            agent_id=self._agent_id,
            content=messages
        ).history

        while not stop:
            logger.debug(f"Context: {context}")

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
                reasoning_effort=reasoning_effort
            )
            
            async for chunk in response_generator:
                if chunk.finish_reason == "tool_calls" and chunk.tool_calls:
                    tool_calls: list[dict[str, Any]] = [tool_call.model_dump() for tool_call in chunk.tool_calls] if chunk.tool_calls else []
                    # Add the llm tool call request to the context
                    context = self.context_store.add_context(
                        thread_id=thread_id,
                        agent_id=self._agent_id,
                        content={
                            "role": "assistant",
                            "tool_calls": tool_calls,
                            "content": str(chunk.content),
                        }
                    ).history

                    yield OpenAgentStreamingResponse(
                        role=chunk.role,
                        content=str(chunk.content) if not isinstance(chunk.content, (BaseModel, type(None))) else chunk.content,
                        tool_calls=chunk.tool_calls,
                        usage=chunk.usage,
                    )

                    logger.debug(f"Context: {context}")

                    # Handle the tool call request and get the final response with tool results
                    tool_response = await self._tool_handler.async_handle_tool_request(
                        tool_calls=chunk.tool_calls,
                    )

                    yield OpenAgentStreamingResponse(
                        role="tool",
                        tool_results=tool_response.tool_results,
                    )

                    logger.debug(f"Tool Messages in Execute: {tool_response.tool_messages}")

                    context = self.context_store.extend_context(
                        thread_id=thread_id,
                        agent_id=self._agent_id,
                        content=[
                            tool_message.model_dump() 
                            for tool_message in tool_response.tool_messages
                        ] 
                        if tool_response.tool_messages else []
                    ).history
                    
                    logger.debug(f"Context in Stream Execute: {context}")

                elif chunk.finish_reason == "stop":
                    logger.debug(f"Final Chunk: {chunk}")
                    if chunk.content:
                        context = self.context_store.add_context(
                            thread_id=thread_id,
                            agent_id=self._agent_id,
                            content={
                                "role": "assistant",
                                "content": str(chunk.content),
                            }
                        ).history
                        logger.debug(f"Context: {context}")
                        yield chunk
                        stop = True
                else:
                    yield chunk
