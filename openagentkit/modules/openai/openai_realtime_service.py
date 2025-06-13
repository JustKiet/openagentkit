from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional, AsyncGenerator, cast
from openai.types.beta.realtime import *

from openagentkit.core.models.responses import OpenAgentStreamingResponse
from openagentkit.core.models.responses import UsageResponse, PromptTokensDetails, CompletionTokensDetails
from openagentkit.core.interfaces import AsyncBaseLLMModel
from openagentkit.core.handlers.tools.tool_wrapper import Tool
from openagentkit.modules.openai import OpenAIAudioVoices
import os
from openai._types import NOT_GIVEN
from openagentkit.core.handlers.tools.tool_handler import ToolHandler
from loguru import logger
import asyncio

class OpenAIRealtimeService(AsyncBaseLLMModel):
    def __init__(self, 
                 client: Optional[AsyncOpenAI] = None,
                 model: str = "gpt-4o-mini-realtime-preview",
                 voice: OpenAIAudioVoices = "alloy",
                 system_message: Optional[str] = None,
                 tools: Optional[List[Tool]] = None,
                 timeout: float = 60.0,
                 api_key: Optional[str] = os.getenv("OPENAI_API_KEY"),
                 temperature: Optional[float] = 0.8,
                 max_tokens: Optional[int] = None,
                 top_p: Optional[float] = None):
        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        if client is None:
            if api_key is None:
                raise ValueError("No API key provided. Please set the OPENAI_API_KEY environment variable or pass it as an argument.")
            self._client = AsyncOpenAI(
                api_key=api_key,
            )
        else:
            self._client = client

        self._model = model
        self._voice: OpenAIAudioVoices = voice
        self._system_message = system_message
        self._tools = tools
        self._tool_handler = ToolHandler(
            tools=tools, llm_provider="openai", schema_type="OpenAIRealtime"
        )
        self.timeout = timeout
        self._api_key = api_key
        self.connection = None
        self._event_queue = asyncio.Queue()
        self._response_queue = asyncio.Queue()
        self._is_connected = False

    def clone(self) -> 'OpenAIRealtimeService':
        return OpenAIRealtimeService(
            client=self._client,
            model=self._model,
            voice=self._voice,
            system_message=self._system_message,
            tools=self._tools,
            api_key=self._api_key,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            top_p=self._top_p
        )

    def _handle_session_event(self, event: RealtimeServerEvent):
        """Handle session-related events"""
        if not event.type.split(".")[0] == "session":
            raise ValueError("Can only handle 'session.*' events!")
        
        match event.type:
            case "session.created":
                logger.info(f"Session created: {event.session}")
            case "session.updated":
                logger.info(f"Session updated: {event.session}")
            case _:
                logger.warning(f"Unhandled session event type: {event.type}")
                return None

    def _handle_conversation_event(self, event: RealtimeServerEvent):
        if not event.type.split(".")[0] == "conversation":
            raise ValueError("Can only handle 'conversation.*' events!")
        match event.type:
            case "conversation.created":
                logger.info(f"{event}")
            case "conversation.item.created":
                logger.info(f"{event}")
            case "conversation.item.input_audio_transcription.completed":
                logger.info(f"{event}")
                return OpenAgentStreamingResponse(
                    role="user",
                    index=event.content_index,
                    finish_reason="transcription",
                    content=event.transcript,
                )
            case "conversation.item.input_audio_transcription.delta":
                logger.info(f"{event}")
            case "conversation.item.input_audio_transcription.failed":
                logger.info(f"{event}")
            case "conversation.item.truncated":
                logger.info(f"{event}")
            case "conversation.item.deleted":
                logger.info(f"{event}")
            case "conversation.item.retrieved":
                logger.info(f"{event}")
            case _:
                logger.warning(f"Unhandled conversation event type: {event.type}")
    
    def _handle_input_audio_buffer_event(self, event: RealtimeServerEvent):
        if not event.type.split(".")[0] == "input_audio_buffer":
            raise ValueError("Can only handle 'input_audio_buffer.*' events!")
        
        match event.type:
            case "input_audio_buffer.committed":
                logger.info(f"{event}")
            case "input_audio_buffer.cleared":
                logger.info(f"{event}")
            case "input_audio_buffer.speech_started":
                logger.info(f"{event}")
            case "input_audio_buffer.speech_stopped":
                logger.info(f"{event}")
            case _:
                logger.warning(f"Unhandled input audio buffer event type: {event.type}")

    def _handle_response_event(self, event: RealtimeServerEvent):
        if not event.type.split(".")[0] == "response":
            raise ValueError("Can only handle 'response.*' events!")

        match event.type:
            case "response.created":
                logger.info(f"{event}")
            case "response.done":
                logger.info(f"{event}")
                
                message = ""
                tool_calls = []
                if event.response.output is None:
                    raise ValueError("No output in response event")
                
                for i in range(len(event.response.output)):
                    if event.response.output[i].type == "function_call":
                        tool_calls.append(
                            {
                                "id": event.response.output[i].call_id,
                                "type": "function",
                                "function": {
                                    "name": event.response.output[i].name,
                                    "arguments": event.response.output[i].arguments,
                                }
                            }
                        )
                    elif event.response.output[i].type == "message":
                        message = event.response.output[i].content[0].text

                return OpenAgentStreamingResponse(
                    role="assistant",
                    content=message,
                    tool_calls=tool_calls,
                    finish_reason="tool_calls" if tool_calls else "stop",
                    usage=UsageResponse(
                        prompt_tokens=event.response.usage.input_tokens,
                        completion_tokens=event.response.usage.output_tokens,
                        total_tokens=event.response.usage.total_tokens,
                        prompt_tokens_details=PromptTokensDetails(
                            cached_tokens=event.response.usage.input_token_details.cached_tokens,
                            audio_tokens=event.response.usage.input_token_details.audio_tokens,
                        ), 
                        completion_tokens_details=CompletionTokensDetails(
                            audio_tokens=event.response.usage.output_token_details.audio_tokens,
                            reasoning_tokens=0,
                            accepted_prediction_tokens=0,
                            rejected_prediction_tokens=0, # TODO: This is missing a bunch of fields
                        )   
                    )
                )
            case "response.output_item.added":
                logger.info(f"{event}")
            case "response.output_item.done":
                logger.info(f"{event}")
            case "response.content_part.added":
                logger.info(f"{event}")
            case "response.content_part.done":
                logger.info(f"{event}")
            case "response.text.delta":
                logger.info(f"{event}")
            case "response.text.done":
                logger.info(f"{event}")
            case "response.audio_transcript.delta":
                return OpenAgentStreamingResponse(
                    role="assistant",
                    index=event.content_index,
                    delta_content=event.delta,    
                )
            case "response.audio_transcript.done":
                logger.info(f"{event}")
                return OpenAgentStreamingResponse(
                    role="assistant",
                    content=event.transcript,    
                )
            case "response.audio.delta":
                return OpenAgentStreamingResponse(
                    role="assistant",
                    index=event.content_index,
                    delta_audio=event.delta,    
                )
            case "response.audio.done":
                logger.info(f"{event}")
            case "response.function_call_arguments.delta":
                logger.info(f"{event}")
            case "response.function_call_arguments.done":
                logger.info(f"{event}")
            case _:
                logger.warning(f"Unhandled response event type: {event.type}")

    def _handle_error_event(self, event: RealtimeServerEvent):
        """Handle error events"""
        if event.type != "error":
            raise ValueError("Can only handle 'error' events!")
        
        logger.error(f"Error received: {event.error}")
        return event.error

    def realtime_event_handler(self, event: RealtimeServerEvent):
        """Handle all realtime events"""
        try:
            subtype = event.type.split(".")[0]
            
            match subtype:
                case "session":
                    return self._handle_session_event(event)
                case "error":
                    return self._handle_error_event(event)
                case "conversation":
                    return self._handle_conversation_event(event)
                case "input_audio_buffer":
                    return self._handle_input_audio_buffer_event(event)
                case "response":
                    return self._handle_response_event(event)
                case "transcription_session":
                    logger.warning(f"Unhandled transcription session event: {event.type}")
                    return None
                case "rate_limits":
                    logger.warning(f"Unhandled rate limits event: {event.type}")
                    logger.warning(f"Rate limits Event: {event}")
                    return None
                case _:
                    logger.warning(f"Unknown event type: {event.type}")
                    return None
        except Exception as e:
            logger.error(f"Error handling event {event.type}: {str(e)}")
            return None
    
    async def _event_listener(self):
        """Continuously listen for events from the WebSocket connection"""
        try:
            async for event in self.connection:
                if event.type != "response.audio.delta":
                    logger.debug(f"Received event: {event.type}")
                await self._event_queue.put(event)
        except Exception as e:
            logger.error(f"Error in event listener: {str(e)}")
            self._is_connected = False
            raise

    async def _event_processor(self):
        """Process events from the queue and handle them appropriately"""
        while self._is_connected:
            try:
                event = await self._event_queue.get()
                response = self.realtime_event_handler(event=event)
                if response:
                    await self._response_queue.put(response)
            except Exception as e:
                logger.error(f"Error processing event: {str(e)}")
                continue

    async def start_up(self):
        """Establish WebSocket connection and start event processing"""
        try:
            logger.info(f"Attempting to connect to Realtime API with model: {self._model}")
            async with self._client.beta.realtime.connect(
                model=self._model
            ) as conn:
                self.connection = conn
                self._is_connected = True
                logger.info("Successfully established WebSocket connection")
                
                # Start event listener and processor
                event_listener_task = asyncio.create_task(self._event_listener())
                event_processor_task = asyncio.create_task(self._event_processor())
                
                # Update session configuration
                session_config = {
                    "modalities": ["text", "audio"],
                    "instructions": self._system_message,
                    "input_audio_transcription": {
                        "model": "gpt-4o-transcribe",
                    },
                    "voice": self._voice,
                    "model": self._model,
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                        "create_response": True,
                    },
                    "input_audio_noise_reduction": {
                        "type": "near_field"
                    },
                    "tools": self._tool_handler.tools,
                    "temperature": self._temperature,
                }
                
                logger.debug(f"Updating session with config: {session_config}")
                await conn.session.update(session=session_config)
                logger.info("Successfully updated session configuration")

                # Keep the connection alive
                try:
                    while self._is_connected:
                        await asyncio.sleep(1)
                except asyncio.CancelledError:
                    logger.info("Connection task cancelled")
                finally:
                    self._is_connected = False
                    event_listener_task.cancel()
                    event_processor_task.cancel()
                    await asyncio.gather(event_listener_task, event_processor_task, return_exceptions=True)

        except Exception as e:
            logger.error(f"Failed to start up Realtime API connection: {str(e)}")
            self._is_connected = False
            raise

    async def model_stream(self, 
                           messages: List[Dict[str, str]], 
                           tools: Optional[List[Dict[str, Any]]] = None, 
                           temperature: Optional[float] = None, 
                           max_tokens: Optional[int] = None, 
                           top_p: Optional[float] = None,
                           audio: Optional[bool] = False,
                           **kwargs):
        """Send a message and yield responses"""
        if not self._is_connected:
            raise RuntimeError("Not connected to Realtime API")

        try:
            message = messages[-1]

            if audio:
                content_payload = ConversationItemContentParam(
                    type="input_audio",
                    audio=message["content"]
                )
            else:
                content_payload = ConversationItemContentParam(
                    type="input_text",
                    text=message["content"]
                )

            await self.connection.conversation.item.create(
                item=ConversationItemParam(
                    type="message",
                    role=message["role"],
                    content=[
                        content_payload 
                    ]
                )
            )

            await self.connection.response.create()

            logger.info("Response created")
            stop = False
            # Yield responses as they come in
            while self._is_connected:
                try:
                    response = await asyncio.wait_for(self._response_queue.get(), timeout=self.timeout)
                    response = cast(OpenAgentStreamingResponse, response)
                    if response.tool_calls:
                        logger.debug(f"Tool calls detected: {response.tool_calls}")
                        
                        tool_response = await self._tool_handler.async_handle_tool_request(
                            response=response,
                        )

                        yield OpenAgentStreamingResponse(
                            role="tool",
                            tool_results=tool_response.tool_results
                        )

                        logger.debug(f"Tool response received: {tool_response}")

                        await self.connection.conversation.item.create(
                            item=ConversationItemParam(
                                type="function_call_output",
                                call_id=tool_response.tool_messages[-1].tool_call_id,
                                output=tool_response.tool_messages[-1].content,
                            )
                        )

                        await self.connection.response.create()
                    else:
                        yield response
                except asyncio.TimeoutError:
                    logger.warning("No response received within timeout period")
                    break

        except Exception as e:
            logger.error(f"Error in model_generate: {str(e)}")
            raise

    async def model_generate(self, 
                           messages: List[Dict[str, str]], 
                           tools: Optional[List[Dict[str, Any]]], 
                           temperature: Optional[float] = None, 
                           max_tokens: Optional[int] = None, 
                           top_p: Optional[float] = None
                           ) -> AsyncGenerator[OpenAgentStreamingResponse, None]:
        pass

    async def get_history(self) -> List[Dict[str, Any]]:
        return None