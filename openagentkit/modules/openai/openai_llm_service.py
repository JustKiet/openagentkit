from typing import Any, Dict, List, Optional, Union, Iterable, cast
from openai import OpenAI
from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDeltaToolCall
from openai.types.chat.chat_completion_stream_options_param import ChatCompletionStreamOptionsParam
from openai.types.chat.chat_completion_audio_param import ChatCompletionAudioParam

from pydantic import BaseModel
from openagentkit.core.handlers.tools.tool_handler import ToolHandler
from openagentkit.core.interfaces import BaseLLMModel
from openagentkit.core.models.responses import (
    OpenAgentStreamingResponse, 
    OpenAgentResponse, 
    UsageResponse, 
    PromptTokensDetails, 
    CompletionTokensDetails, 
)
from openagentkit.core.handlers.tools.tool_wrapper import Tool
from openagentkit.core.models.responses.tool_response import *
from openagentkit.core.models.responses.audio_response import AudioResponse
from openagentkit.modules.openai import OpenAIAudioFormats, OpenAIAudioVoices
from typing import Generator
import os
from loguru import logger

class OpenAILLMService(BaseLLMModel):
    def __init__(
        self, 
        client: Optional[OpenAI] = None,
        model: str = "gpt-4o-mini",
        tools: Optional[List[Tool]] = None,
        api_key: Optional[str] = os.getenv("OPENAI_API_KEY"),
        temperature: Optional[float] = 0.3,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> None:
        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        self._tool_handler = ToolHandler(
            tools=tools, llm_provider="openai", schema_type="OpenAI"
        )
        
        if client is None:
            if api_key is None:
                raise ValueError("No API key provided. Please set the OPENAI_API_KEY environment variable or pass it as an argument.")
            self._client = OpenAI(
                api_key=api_key,
            )
        else:
            self._client = client

        self._tools = tools
        self._model = model
        self._api_key = api_key

    @property
    def client(self) -> OpenAI | None:
        """
        Get the OpenAI client.

        Returns:
            The OpenAI client.
        """
        return self._client
    
    @property
    def api_key(self) -> str | None:
        """
        Get the API key.

        Returns:
            The API key.
        """
        return self._api_key

    @property
    def tool_handler(self) -> ToolHandler:
        """
        Get the tool handler.

        Returns:
            The tool handler.
        """
        return self._tool_handler
    
    @tool_handler.setter
    def tool_handler(self, value: ToolHandler) -> None:
        """
        Set the tool handler.

        Args:
            value: The tool handler to set.
        """
        self._tool_handler = value

    @property
    def model(self) -> str:
        """
        Get the model name.

        Returns:
            The model name.
        """
        return self._model
    
    # Property to access tools from the tool handler
    @property
    def tools(self):
        """
        Get the tools from the tool handler.

        Returns:
            The tools from the tool handler.
        """
        return self._tool_handler.tools
    
    def clone(self) -> 'OpenAILLMService':
        """
        Clone the LLM model instance.

        Returns:
            A clone of the LLM model instance.
        """
        return OpenAILLMService(
            client=self._client,
            model=self._model,
            tools=self._tools,
            api_key=self._api_key,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            top_p=self._top_p,
        )
    
    def _handle_client_request(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[Union[List[Dict[str, Any]], NotGiven]] = None,
        response_schema: Union[type[BaseModel], NotGiven] = NOT_GIVEN,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        audio: Optional[bool] = False,
        audio_format: Optional[OpenAIAudioFormats] = "pcm16",
        audio_voice: Optional[OpenAIAudioVoices] = None,
        **kwargs: Any
    ) -> OpenAgentResponse:
        """
        Handle the client request.

        Args:
            messages: The messages to send to the model.
            tools: The tools to use in the response.
            response_schema: The schema to use in the response.
            temperature: The temperature to use in the response.
            max_tokens: The max tokens to use in the response.
            top_p: The top p to use in the response.

        Returns:
            An OpenAgentResponse object.
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

        if tools is None:
            tools = NOT_GIVEN

        if response_schema is NOT_GIVEN or isinstance(response_schema, NotGiven):
            # Handle the client request without response schema
            client_response = self._client.chat.completions.create(
                model=self._model,
                messages=messages, # type: ignore
                tools=tools, # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                modalities=["text", "audio"] if audio else ["text"],
                audio=ChatCompletionAudioParam(
                    format=audio_format,
                    voice=audio_voice,
                ) if audio and audio_format and audio_voice else None,
            )
            
            response_message = client_response.choices[0].message

            # Create the response object
            response = OpenAgentResponse(
                role=response_message.role,
                content=response_message.content,
                tool_calls=[
                    ToolCallResponse(
                        id=tool_call.id,
                        type=tool_call.type,
                        function=ToolCallFunction(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        ),
                    )
                    for tool_call in response_message.tool_calls
                ] if response_message.tool_calls else None,
                refusal=response_message.refusal,
                audio=AudioResponse(
                    id=response_message.audio.id,
                    data=response_message.audio.data,
                    transcription=response_message.audio.transcript,
                ) if response_message.audio else None,
            )

        else:
            # Handle the client request with response schema
            client_response = self._client.beta.chat.completions.parse(
                model=self._model,
                messages=messages, # type: ignore
                tools=tools, # type: ignore
                response_format=response_schema, # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )

            response_message = client_response.choices[0].message

            # Create the response object
            response = OpenAgentResponse(
                role=response_message.role,
                content=response_message.content,
                tool_calls=[
                    ToolCallResponse(
                        id=tool_call.id,
                        type=tool_call.type,
                        function=ToolCallFunction(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        ),
                    )
                    for tool_call in response_message.tool_calls
                ] if response_message.tool_calls else None,
                refusal=response_message.refusal,
                audio=AudioResponse(
                    id=response_message.audio.id,
                    data=response_message.audio.data,
                    transcription=response_message.audio.transcript,
                ) if response_message.audio else None,
            )

        # Add usage info to the response
        response.usage = UsageResponse(
            prompt_tokens=client_response.usage.prompt_tokens,
            completion_tokens=client_response.usage.completion_tokens,
            total_tokens=client_response.usage.total_tokens,
            prompt_tokens_details=PromptTokensDetails(
                cached_tokens=client_response.usage.prompt_tokens_details.cached_tokens,
                audio_tokens=client_response.usage.prompt_tokens_details.audio_tokens,
            ) if client_response.usage.prompt_tokens_details else None,
            completion_tokens_details=CompletionTokensDetails(
                reasoning_tokens=client_response.usage.completion_tokens_details.reasoning_tokens,
                audio_tokens=client_response.usage.completion_tokens_details.audio_tokens,
                accepted_prediction_tokens=client_response.usage.completion_tokens_details.accepted_prediction_tokens,
                rejected_prediction_tokens=client_response.usage.completion_tokens_details.rejected_prediction_tokens,
            ) if client_response.usage.completion_tokens_details else None,
        ) if client_response.usage else None
        
        return response
    
    def _handle_client_stream(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_schema: Optional[type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        audio: Optional[bool] = False,
        audio_format: Optional[OpenAIAudioFormats] = "pcm16",
        audio_voice: Optional[OpenAIAudioVoices] = None,
        **kwargs: Any
    ) -> Generator[OpenAgentStreamingResponse, None, None]:
        """
        Handle the client stream.

        Args:
            messages: The messages to send to the model.
            tools: The tools to use in the response.
            response_schema: The schema to use in the response. **(not implemented yet)**
            temperature: The temperature to use in the response.
            max_tokens: The max tokens to use in the response.
            top_p: The top p to use in the response.
            audio: Whether to include audio in the response.
            audio_format: The audio format to use in the response.
            audio_voice: The audio voice to use in the response.

        Returns:
            A Generator[OpenAgentStreamingResponse, None] object.
        """
        # TODO: THIS IS A PLACEHOLDER FOR NOW, WE NEED TO IMPLEMENT THE STREAMING FOR THE RESPONSE SCHEMA
        if isinstance(response_schema, BaseModel):
            raise ValueError("Response schema is not supported for streaming")
        
        temperature = kwargs.get("temperature", temperature)
        if temperature is None:
            temperature = self.temperature

        max_tokens = kwargs.get("max_tokens", max_tokens)
        if max_tokens is None:
            max_tokens = self.max_tokens

        top_p = kwargs.get("top_p", top_p)
        if top_p is None:
            top_p = self.top_p

        if tools is None:
            tools = self.tools

        if audio and not audio_format:
            raise ValueError("Audio format is required when audio is True")
        
        if audio and not audio_voice:
            raise ValueError("Audio voice is required when audio is True")
        
        if not response_schema:
            client_stream = self._client.chat.completions.create( # type: ignore
                model=self._model,
                messages=messages, # type: ignore
                tools=tools, # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=True,
                stream_options=ChatCompletionStreamOptionsParam(
                    include_usage=True,
                ),
                modalities=["text", "audio"] if audio else ["text"],
                audio=ChatCompletionAudioParam(
                    format=audio_format,
                    voice=audio_voice,
                ) if audio and audio_format and audio_voice else None,
            )

            client_stream = cast(Iterable[ChatCompletionChunk], client_stream)

            # Initialize variables to store the final tool calls, content, and chunk
            final_tool_calls: dict[int, ChoiceDeltaToolCall] = {}
            final_content = ""
            final_chunk = None

            for chunk in client_stream:
                final_chunk = chunk  # Store the last chunk for usage info
                
                # If the chunk is empty, skip it
                if not chunk.choices:
                    continue
                    
                # If the chunk has content, yield it
                if chunk.choices[0].delta.content is not None:
                    final_content += chunk.choices[0].delta.content
                    yield OpenAgentStreamingResponse(
                        role="assistant",
                        delta_content=chunk.choices[0].delta.content,
                        finish_reason=chunk.choices[0].finish_reason,
                    )

                # If the chunk has tool calls, add them to the final tool calls
                if chunk.choices[0].delta.tool_calls:
                    for tool_call in chunk.choices[0].delta.tool_calls:
                        index = tool_call.index

                        if index not in final_tool_calls:
                            final_tool_calls[index] = tool_call

                        if tool_call.function and tool_call.function.arguments:
                            final_tool_calls[index].function.arguments += tool_call.function.arguments # type: ignore

                # Handle audio chunks (if available)
                if chunk.choices[0].delta.model_dump().get("audio") is not None:
                    if chunk.choices[0].delta.model_dump().get("audio").get("data") is not None: # type: ignore
                        yield OpenAgentStreamingResponse(
                            role="assistant",
                            delta_audio=chunk.choices[0].delta.model_dump().get("audio").get("data"), # type: ignore
                            finish_reason=chunk.choices[0].finish_reason,
                        )
                    
                    if chunk.choices[0].delta.model_dump().get("audio").get("transcript") is not None: # type: ignore
                        final_content += chunk.choices[0].delta.model_dump().get("audio").get("transcript") # type: ignore
                        yield OpenAgentStreamingResponse(
                            role="assistant",
                            delta_content=chunk.choices[0].delta.model_dump().get("audio").get("transcript"), # type: ignore
                            finish_reason=chunk.choices[0].finish_reason,
                        )
            
            tool_calls = list(final_tool_calls.values())

            # After the stream is done, yield the final response with usage info if available
            if final_chunk and hasattr(final_chunk, 'usage') and final_chunk.usage is not None:
                yield OpenAgentStreamingResponse(
                    role="assistant",
                    content=final_content if final_content else None,
                    finish_reason="tool_calls" if final_tool_calls else "stop",
                    tool_calls=[ToolCallResponse(**tool_call.model_dump()) for tool_call in tool_calls],
                    usage=UsageResponse(
                        prompt_tokens=final_chunk.usage.prompt_tokens,
                        completion_tokens=final_chunk.usage.completion_tokens,
                        total_tokens=final_chunk.usage.total_tokens,
                        prompt_tokens_details=PromptTokensDetails(
                            cached_tokens=final_chunk.usage.prompt_tokens_details.cached_tokens,
                            audio_tokens=final_chunk.usage.prompt_tokens_details.audio_tokens,
                        ) if final_chunk.usage.prompt_tokens_details else None,
                        completion_tokens_details=CompletionTokensDetails(
                            reasoning_tokens=final_chunk.usage.completion_tokens_details.reasoning_tokens,
                            audio_tokens=final_chunk.usage.completion_tokens_details.audio_tokens,
                            accepted_prediction_tokens=final_chunk.usage.completion_tokens_details.accepted_prediction_tokens,
                            rejected_prediction_tokens=final_chunk.usage.completion_tokens_details.rejected_prediction_tokens,
                        ) if final_chunk.usage.completion_tokens_details else None,
                    ),
                )
            else:
                logger.warning("Final chunk or usage is None")
                
                yield OpenAgentStreamingResponse(
                    role="assistant",
                    content=final_content,
                    finish_reason="tool_calls" if final_tool_calls else "stop",
                    tool_calls=[ToolCallResponse(**tool_call.model_dump()) for tool_call in tool_calls],
                )

        
        # TODO: Handle the case with response schema (not working)
        else:
            with self._client.beta.chat.completions.stream(
                model=self._model,
                messages=messages, # type: ignore
                tools=tools, # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream_options={"include_usage": True},
                response_format=response_schema,
            ) as client_stream:
                for event in client_stream:
                    if event.type == "content.delta":
                        if event.parsed is not None:
                                # Print the parsed data as JSON
                                print("content.delta parsed:", event.parsed)
                                break
                        elif event.type == "content.done": # type: ignore
                            print("content.done")
                            break
                        elif event.type == "error": # type: ignore
                            print("Error in stream:", event.error)
                            break
        
    def model_generate(
        self, 
        messages: List[Dict[str, str]],
        response_schema: Optional[type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        audio: Optional[bool] = False,
        audio_format: Optional[OpenAIAudioFormats] = "pcm16",
        audio_voice: Optional[OpenAIAudioVoices] = None,
        **kwargs: Any
    ) -> OpenAgentResponse:
        """
        Generate a response from the model.
        
        Args:
            messages: The messages to send to the model.
            tools: The tools to use in the response.
            response_schema: The schema to use in the response.
            temperature: The temperature to use in the response.
            max_tokens: The max tokens to use in the response.
            top_p: The top p to use in the response.
            audio: Whether to include audio in the response.
            audio_format: The audio format to use in the response.
            audio_voice: The audio voice to use in the response.

        Returns:
            An OpenAgentResponse object.

        Example:
        ```python
        from openagentkit.tools import duckduckgo_search_tool
        from openagentkit.modules.openai import AsyncOpenAILLMService

        llm_service = AsyncOpenAILLMService(client, tools=[duckduckgo_search_tool])
        response = await llm_service.model_generate(messages=[{"role": "user", "content": "What is TECHVIFY?"}])
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

        if tools is None:
            tools = self.tools
            
        #logger.info(f"Tools: {tools}")

        # Handle the client request
        response = self._handle_client_request(
            messages=messages, 
            response_schema=response_schema if response_schema else NOT_GIVEN, 
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            audio=audio,
            audio_format=audio_format,
            audio_voice=audio_voice,
        )
        
        if response.tool_calls:
            # Extract tool_calls arguments using the tool handler
            tool_calls = self._tool_handler.parse_tool_args(response)
            
            # Update the response with the parsed tool calls
            response.tool_calls = tool_calls
        
        return response

    def model_stream(
        self,
        messages: List[Dict[str, str]],
        response_schema: Optional[type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        audio: Optional[bool] = False,
        audio_format: Optional[OpenAIAudioFormats] = "pcm16",
        audio_voice: Optional[OpenAIAudioVoices] = "alloy",
        **kwargs: Any
    ) -> Generator[OpenAgentStreamingResponse, None, None]:
        """
        Generate a response from the model.

        Args:
            messages: The messages to send to the model.
            tools: The tools to use in the response.
            response_schema: The schema to use in the response. **(not implemented yet)**
            temperature: The temperature to use in the response.
            max_tokens: The max tokens to use in the response.
            top_p: The top p to use in the response.
            audio: Whether to include audio in the response.
            audio_format: The audio format to use in the response.
            audio_voice: The audio voice to use in the response.

        Returns:
            An AsyncGenerator[OpenAgentStreamingResponse, None] object.
        """
        # TODO: Handle the case with response schema (not working)
        if isinstance(response_schema, BaseModel):
            raise ValueError("Response schema is not supported for streaming")
        
        temperature = kwargs.get("temperature", temperature)
        if temperature is None:
            temperature = self.temperature

        max_tokens = kwargs.get("max_tokens", max_tokens)
        if max_tokens is None:
            max_tokens = self.max_tokens

        top_p = kwargs.get("top_p", top_p)
        if top_p is None:
            top_p = self.top_p

        if tools is None:
            tools = self.tools

        generator = self._handle_client_stream(
            messages=messages, 
            tools=tools, 
            response_schema=response_schema, 
            temperature=temperature, 
            max_tokens=max_tokens, 
            top_p=top_p,
            audio=audio,
            audio_format=audio_format,
            audio_voice=audio_voice,
        )

        for chunk in generator:
            if chunk.tool_calls:
                # Extract tool_calls arguments using the tool handler
                tool_calls = self._tool_handler.parse_tool_args(chunk)
                
                # Update the chunk with the parsed tool calls
                chunk.tool_calls = tool_calls
            yield chunk