from typing import Any, Callable, Dict, List, Optional, Union
from openai import AsyncOpenAI
from openai._types import NOT_GIVEN
from pydantic import BaseModel
from openagentkit.handlers.tool_handler import ToolHandler
from openagentkit.interfaces import AsyncBaseLLMModel
from openagentkit.models.responses import (
    OpenAIStreamingResponse, 
    OpenAgentResponse, 
    UsageResponse, 
    PromptTokensDetails, 
    CompletionTokensDetails
)
from typing import AsyncGenerator
import asyncio
import os

class AsyncOpenAILLMService(AsyncBaseLLMModel):
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
        # Create an instance of ToolHandler instead of inheriting from it
        self._tool_handler = ToolHandler(tools=tools)
        
        self._client = client
        self._model = model
        self._system_message = system_message
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._context_history = [
            {
                "role": "system",
                "content": self._system_message,
            }
        ]
    
    # Property to access tools from the tool handler
    @property
    def tools(self):
        return self._tool_handler.tools
        
    async def define_system_message(self) -> str:
        return self._system_message
    
    async def _handle_client_request(self,
                                     messages: List[Dict[str, str]],
                                     tools: Optional[List[Dict[str, Any]]],
                                     response_schema: BaseModel = NOT_GIVEN,
                                     ) -> OpenAgentResponse:


        if response_schema is NOT_GIVEN:
            # Handle the client request without response schema
            client_response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=tools,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    top_p=self._top_p,
            )
            
            response_message = client_response.choices[0].message

            # Create the response object
            response = OpenAgentResponse(
                role=response_message.role,
                content=response_message.content,
                tool_calls=response_message.tool_calls,
                refusal=response_message.refusal,
                audio=response_message.audio,
            )

        else:
            # Handle the client request with response schema
            client_response = await self._client.beta.chat.completions.parse(
                model=self._model,
                messages=messages,
                tools=tools,
                response_format=response_schema,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                top_p=self._top_p,
            )

            response_message = client_response.choices[0].message

            # Create the response object    
            response = OpenAgentResponse(
                role=response_message.role,
                content=response_message.parsed,
                tool_calls=response_message.tool_calls,
                refusal=response_message.refusal,
                audio=response_message.audio,
            )

        # Add usage info to the response
        response.usage = UsageResponse(
            prompt_tokens=client_response.usage.prompt_tokens,
            completion_tokens=client_response.usage.completion_tokens,
            total_tokens=client_response.usage.total_tokens,
            prompt_tokens_details=PromptTokensDetails(
                cached_tokens=client_response.usage.prompt_tokens_details.cached_tokens,
                audio_tokens=client_response.usage.prompt_tokens_details.audio_tokens,
            ),
            completion_tokens_details=CompletionTokensDetails(
                reasoning_tokens=client_response.usage.completion_tokens_details.reasoning_tokens,
                audio_tokens=client_response.usage.completion_tokens_details.audio_tokens,
                accepted_prediction_tokens=client_response.usage.completion_tokens_details.accepted_prediction_tokens,
                rejected_prediction_tokens=client_response.usage.completion_tokens_details.rejected_prediction_tokens,
            ),
        )
        
        return response
    
    async def _handle_client_stream(self,
                                    messages: List[Dict[str, str]],
                                    tools: Optional[List[Dict[str, Any]]] = None,
                                    response_schema: BaseModel = NOT_GIVEN) -> AsyncGenerator[OpenAIStreamingResponse, None]:
        
        if tools is None:
            tools = self.tools
        
        if response_schema is NOT_GIVEN:
            client_stream = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                top_p=self._top_p,
                stream=True,
                stream_options={"include_usage": True},
            )

            final_tool_calls = {}
            final_content = ""
            final_chunk = None
            
            async for chunk in client_stream:
                final_chunk = chunk  # Store the last chunk for usage info
                
                # If the chunk is empty, skip it
                if not chunk.choices:
                    continue
                    
                # If the chunk has content, yield it
                if chunk.choices[0].delta.content is not None:
                    final_content += chunk.choices[0].delta.content
                    yield OpenAIStreamingResponse(
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

                        final_tool_calls[index].function.arguments += tool_call.function.arguments
            
            # After the stream is done, yield the final response with usage info if available
            if final_chunk and hasattr(final_chunk, 'usage'):
                yield OpenAIStreamingResponse(
                    role="assistant",
                    content=final_content,  # Empty content for the final usage info
                    finish_reason="tool_calls" if final_tool_calls else "stop",
                    tool_calls=list(final_tool_calls.values()),
                    usage=UsageResponse(
                        prompt_tokens=final_chunk.usage.prompt_tokens,
                        completion_tokens=final_chunk.usage.completion_tokens,
                        total_tokens=final_chunk.usage.total_tokens,
                        prompt_tokens_details=PromptTokensDetails(
                            cached_tokens=final_chunk.usage.prompt_tokens_details.cached_tokens,
                            audio_tokens=final_chunk.usage.prompt_tokens_details.audio_tokens,
                        ),
                        completion_tokens_details=CompletionTokensDetails(
                            reasoning_tokens=final_chunk.usage.completion_tokens_details.reasoning_tokens,
                            audio_tokens=final_chunk.usage.completion_tokens_details.audio_tokens,
                            accepted_prediction_tokens=final_chunk.usage.completion_tokens_details.accepted_prediction_tokens,
                            rejected_prediction_tokens=final_chunk.usage.completion_tokens_details.rejected_prediction_tokens,
                        ),
                    ),
                )
        else:
            with self._client.beta.chat.completions.stream(
                model=self._model,
                messages=messages,
                tools=tools,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                top_p=self._top_p,
                stream=True,
                stream_options={"include_usage": True},
                response_format=response_schema,
            ) as stream:
                for event in stream:
                    if event.type == "content.delta":
                        if event.parsed is not None:
                            # Print the parsed data as JSON
                            print("content.delta parsed:", event.parsed)
                    elif event.type == "content.done":
                        print("content.done")
                    elif event.type == "error":
                        print("Error in stream:", event.error)

            final_completion = stream.get_final_completion()
            print("Final completion:", final_completion)

    async def model_stream(self,
                           messages: List[Dict[str, str]],
                           tools: List[Dict[str, Any]] = None,
                           response_schema: BaseModel = NOT_GIVEN) -> AsyncGenerator[OpenAIStreamingResponse, None]:
        if tools is None:
            tools = self.tools

        generator = self._handle_client_stream(messages, tools, response_schema)

        async for chunk in generator:
            if chunk.tool_calls:
                # Extract tool_calls arguments using the tool handler
                tool_calls = self._tool_handler.parse_tool_args(chunk)
                
                # Update the chunk with the parsed tool calls
                chunk.tool_calls = tool_calls
            yield chunk
        
    async def model_generate(self, 
                       messages: List[Dict[str, str]],
                       tools: Optional[List[Dict[str, Any]]] = None,
                       response_schema: Optional[BaseModel] = NOT_GIVEN) -> OpenAgentResponse:
        """
        Generate a response from the model.
        
        :param messages: The messages to send to the model.
        :param tools: The tools to use in the response.
        :param response_schema: The schema to use in the response.
        :return: An OpenAgentResponse object.

        Example:
        ```python
        from openagentkit.tools import duckduckgo_search_tool
        from openagentkit.modules.openai import AsyncOpenAILLMService

        llm_service = AsyncOpenAILLMService(client, tools=[duckduckgo_search_tool])
        response = await llm_service.model_generate(messages=[{"role": "user", "content": "What is TECHVIFY?"}])
        ```
        """
        if tools is None:
            tools = self.tools
            
        #logger.info(f"Tools: {tools}")

        # Handle the client request
        response = await self._handle_client_request(
            messages=messages, 
            response_schema=response_schema, 
            tools=tools,
        )
            
        # Extract tool_calls arguments using the tool handler
        tool_calls = self._tool_handler.parse_tool_args(response)
        
        # Update the response with the parsed tool calls
        response.tool_calls = tool_calls
        
        return response
        
    async def add_context(self, content: List[dict] | dict):
        if not content:
            return self._context_history
        
        self._context_history.append(content)
        return self._context_history
        
    async def extend_context(self, content: List[dict[str, str]]):
        if not content:
            return self._context_history
        
        self._context_history.extend(content)
        return self._context_history
    
    # Delegate tool call handling to the tool handler
    async def _handle_tool_call(self, tool_name, **tool_args):
        result = self._tool_handler._handle_tool_call(tool_name, **tool_args)
        # Convert result to string if it's not already
        if not isinstance(result, str):
            return str(result)
        return result