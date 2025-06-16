from typing import Any, Dict, List, Optional, Generator
import os
from loguru import logger
from openai._types import NOT_GIVEN
from openai import OpenAI
from openagentkit.core.interfaces.base_agent import BaseAgent
from openagentkit.modules.openai import OpenAILLMService
from openagentkit.core.models.responses import OpenAgentResponse, OpenAgentStreamingResponse
from openagentkit.core.tools.tool_handler import ToolHandler
from openagentkit.core.tools.base_tool import Tool
from openagentkit.modules.openai import OpenAIAudioFormats, OpenAIAudioVoices
from pydantic import BaseModel

class OpenAIAgent(BaseAgent):
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = "gpt-4o-mini",
        system_message: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        api_key: Optional[str] = os.getenv("OPENAI_API_KEY"),
        temperature: Optional[float] = 0.3,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        context_history = kwargs.get("context_history", None)
        super().__init__(system_message=system_message, context_history=context_history)

        self._llm_service = OpenAILLMService(
            client=client,
            model=model,
            tools=tools,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        self._tool_handler = ToolHandler(
            tools=tools
        )

        self._tools = tools

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
    
    def clone(self) -> 'OpenAIAgent':
        """
        Clone the OpenAIAgent object.

        Returns:
            A new OpenAIAgent object with the same parameters.
        """
        return OpenAIAgent(
            client=self._llm_service.client,
            model=self._llm_service.model,
            system_message=self._system_message,
            tools=self._tools,
            api_key=self._llm_service.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )

    def execute(
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
    ) -> Generator[OpenAgentResponse, None, None]:
        """
        Execute the OpenAI model and return an OpenAgentResponse object.

        Args:
            messages (List[Dict[str, str]]): The messages to send to the model.
            tools (Optional[List[Dict[str, Any]]]): The tools to use in the response.
            temperature (Optional[float]): The temperature to use in the response.
            max_tokens (Optional[int]): The maximum number of tokens to use in the response.
            top_p (Optional[float]): The top p to use in the response.
            response_schema (Optional[type[BaseModel]]): The schema to use in the response.
            audio (Optional[bool]): Whether to use audio in the response.
            audio_format (Optional[OpenAIAudioFormats]): The audio format to use in the response.
            audio_voice (Optional[OpenAIAudioVoices]): The audio voice to use in the response.

        Returns:
            An OpenAgentResponse generator.
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
        
        context = self.extend_context(messages)
        
        logger.debug(f"Context: {context}") if debug else None

        stop = False

        while not stop:
            # Take user initial request along with the chat history -> response
            response = self._llm_service.model_generate(
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
                context = self.add_context(
                    {
                        "role": response.role,
                        "content": str(response.content),
                    }
                )

            tool_results: list[Any] = []
            
            if response.tool_calls:
                # Add the tool call request to the context
                tool_calls: list[dict[str, str]] = [tool_call.model_dump() for tool_call in response.tool_calls]

                context = self.add_context(
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

                # Handle tool requests and get the final response with tool results
                tool_response = self._tool_handler.handle_tool_request(
                    tool_calls=response.tool_calls
                )

                yield OpenAgentResponse(
                    role="tool",
                    tool_results=tool_response.tool_results,
                )

                logger.debug(f"Tool Messages in Execute: {tool_response.tool_messages}") if debug else None

                context = self.extend_context([tool_message.model_dump() for tool_message in tool_response.tool_messages] if tool_response.tool_messages else [])

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
                

    def stream_execute(
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
    ) -> Generator[OpenAgentStreamingResponse, None, None]:
        """
        Stream execute the OpenAI model and return an OpenAgentStreamingResponse object.

        Args:
            messages (List[Dict[str, str]]): The messages to send to the model.
            tools (Optional[List[Dict[str, Any]]]): The tools to use in the response.
            response_schema (Optional[type[BaseModel]]): The schema to use in the response.
            temperature (Optional[float]): The temperature to use in the response.
            max_tokens (Optional[int]): The maximum number of tokens to use in the response.
            top_p (Optional[float]): The top p to use in the response.

        Returns:
            An OpenAgentStreamingResponse generator.
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

        context: list[dict[str, Any]] = self.extend_context(messages)

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
            
            for chunk in response_generator:
                if chunk.finish_reason == "tool_calls" and chunk.tool_calls:
                    # Add the llm tool call request to the context
                    context = self.add_context(
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

                    # Handle the tool call request and get the final response with tool results
                    tool_response = self._tool_handler.handle_tool_request(
                        tool_calls=chunk.tool_calls,
                    )

                    yield OpenAgentStreamingResponse(
                        role="tool",
                        tool_results=tool_response.tool_results,
                    )

                    logger.debug(f"Tool Messages in Execute: {tool_response.tool_messages}") if debug else None
                    
                    context = self.extend_context([tool_message.model_dump() for tool_message in tool_response.tool_messages] if tool_response.tool_messages else [])
                    
                    logger.debug(f"Context in Stream Execute: {context}") if debug else None

                elif chunk.finish_reason == "stop":
                    if chunk.content:
                        context = self.add_context(
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