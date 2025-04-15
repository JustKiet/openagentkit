from openai import AsyncOpenAI, AsyncAzureOpenAI
from typing import List, Dict, Any, Optional, AsyncGenerator
from openai.types.beta.realtime import *

from openagentkit.models.payloads.realtime_payload import (
    RealtimeSessionPayload, 
    RealtimeTurnDetectionConfig, 
    RealtimeClientPayload,
)
from openagentkit.models.responses import OpenAgentResponse, OpenAgentStreamingResponse
from openagentkit.interfaces import AsyncBaseLLMModel

import os
from openai._types import NOT_GIVEN, NotGiven
from typing import Callable, Literal
from loguru import logger
import websockets
import asyncio

class OpenAIRealtimeService(AsyncBaseLLMModel):
    def __init__(self, 
                 client: AsyncOpenAI,
                 model: str = "gpt-4o-mini-realtime-preview",
                 voice: Literal["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"] = None,
                 system_message: Optional[str] = None,
                 tools: Optional[List[Callable[..., Any]]] = NOT_GIVEN,
                 api_key: Optional[str] = os.getenv("OPENAI_API_KEY"),
                 temperature: Optional[float] = 0.3,
                 max_tokens: Optional[int] = None,
                 top_p: Optional[float] = None):
        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        self._client = client
        self._model = model
        self._voice = voice
        self._system_message = system_message
        self._tools = tools
        self._api_key = api_key
        self.connection = None
        self._event_queue = asyncio.Queue()
        self._response_queue = asyncio.Queue()
        self._is_connected = False

    def _handle_session_event(self, event: RealtimeServerEvent):
        """Handle session-related events"""
        if not event.type.split(".")[0] == "session":
            raise ValueError("Can only handle 'session.*' events!")
        
        match event.type:
            case "session.created":
                logger.info(f"Session created: {event.session}")
                return event.session
            case "session.updated":
                logger.info(f"Session updated: {event.session}")
                return event.session
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
                return event.transcript
            case "conversation.item.input_audio_transcription.delta":
                logger.info(f"{event}")
            case "conversation.item.input_audio_transcription.failed":
                logger.info(f"{event}")
            case "conversation.item.truncated":
                logger.info(f"{event}")
            case "conversation.item.deleted":
                logger.info(f"{event}")
    
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

    def _handle_response_event(self, event: RealtimeServerEvent):
        if not event.type.split(".")[0] == "response":
            raise ValueError("Can only handle 'response.*' events!")
        
        match event.type:
            case "response.created":
                logger.info(f"{event}")
            case "response.done":
                logger.info(f"{event}")
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
                return event.delta
            case "response.audio_transcript.done":
                return event.transcript
            case "response.audio.delta":
                return event.delta
            case "response.audio.done":
                logger.info(f"{event}")
            case "response.audio.delta":
                logger.info(f"{event}")
            case "response.audio.done":
                logger.info(f"{event}")
            case "response.function_call_arguments.delta":
                logger.info(f"{event}")
            case "response.audio_transcript.done":
                logger.info(f"{event}")

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
                logger.debug(f"Received event: {event}")
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
                    "voice": self._voice,
                    "model": self._model,
                    "turn_detection": {
                        "type": "server_vad",
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 200,
                        "threshold": 0.5,
                        "create_response": True,
                        "interrupt_response": True
                    }
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

    async def model_generate(self, 
                           messages: List[Dict[str, str]], 
                           tools: Optional[List[Dict[str, Any]]], 
                           temperature: Optional[float] = None, 
                           max_tokens: Optional[int] = None, 
                           top_p: Optional[float] = None,
                           **kwargs) -> AsyncGenerator[OpenAgentResponse, None]:
        """Send a message and yield responses"""
        if not self._is_connected:
            raise RuntimeError("Not connected to Realtime API")

        try:
            message = messages[-1]
            logger.info(f"Sending message: {message}")
            
            await self.connection.conversation.item.create(
                item=ConversationItemParam(
                    type="message",
                    role=message["role"],
                    content=[
                        ConversationItemContentParam(
                            type="input_text",
                            text=message["content"]
                        )
                    ]
                )
            )

            await self.connection.response.create()

            # Yield responses as they come in
            while self._is_connected:
                try:
                    response = await asyncio.wait_for(self._response_queue.get(), timeout=30.0)
                    yield response
                except asyncio.TimeoutError:
                    logger.warning("No response received within timeout period")
                    break

        except Exception as e:
            logger.error(f"Error in model_generate: {str(e)}")
            raise

    async def model_stream(self, 
                           messages: List[Dict[str, str]], 
                           tools: Optional[List[Dict[str, Any]]], 
                           temperature: Optional[float] = None, 
                           max_tokens: Optional[int] = None, 
                           top_p: Optional[float] = None
                           ) -> AsyncGenerator[OpenAgentStreamingResponse, None]:
        pass

    async def get_history(self) -> List[Dict[str, Any]]:
        return self._context_history