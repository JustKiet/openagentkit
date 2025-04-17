from typing import Any, Callable, Dict, List, Optional, Union, Generator
from openai import OpenAI
from openai._types import NOT_GIVEN
from pydantic import BaseModel
from openagentkit.core.handlers.tool_handler import ToolHandler
from openagentkit.core.interfaces.base_llm_model import BaseLLMModel
from openagentkit.core.models.responses import (
    OpenAgentResponse, 
    UsageResponse, 
    PromptTokensDetails, 
    CompletionTokensDetails, 
    OpenAgentStreamingResponse
)
import os

class OpenAILLMService(BaseLLMModel):
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
        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            *args,
            **kwargs
        )
        # Create an instance of ToolHandler instead of inheriting from it
        self._tool_handler = ToolHandler(tools=tools)
        
        self._client = client
        self._model = model
        self._system_message = system_message
        self._api_key = api_key
        self._context_history = [
            {
                "role": "system",
                "content": self._system_message,
            }
        ]

    @property
    def model(self) -> str:
        """
        Get the model name.

        Returns:
            The model name.
        """
        return self._model
    
    @property
    def history(self) -> List[Dict[str, Any]]:
        """
        Get the history of the conversation.

        Returns:
            The history of the conversation.
        """
        return self._context_history
    
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
            system_message=self._system_message,
            tools=self.tools,
            api_key=self._api_key,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            top_p=self._top_p,
        )
    
    def _handle_client_request(self,
                              messages: List[Dict[str, str]],
                              tools: Optional[List[Dict[str, Any]]],
                              response_schema: Optional[BaseModel] = NOT_GIVEN,
                              temperature: Optional[float] = None,
                              max_tokens: Optional[int] = None,
                              top_p: Optional[float] = None,
                              **kwargs) -> OpenAgentResponse:
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
            temperature = self._temperature

        max_tokens = kwargs.get("max_tokens", max_tokens)
        if max_tokens is None:
            max_tokens = self._max_tokens

        top_p = kwargs.get("top_p", top_p)
        if top_p is None:
            top_p = self._top_p

        if tools is None:
            tools = self.tools

        if response_schema is NOT_GIVEN:
            # Handle the client request without response schema
            client_response = self._client.chat.completions.create(
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
            client_response = self._client.beta.chat.completions.parse(
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
        
    def model_generate(self, 
                       messages: List[Dict[str, str]],
                       tools: Optional[List[Dict[str, Any]]] = None,
                       response_schema: Optional[BaseModel] = NOT_GIVEN,
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None,
                       top_p: Optional[float] = None,
                       **kwargs) -> OpenAgentResponse:
        """
        Generate a response from the model.
        
        Args:
            messages: The messages to send to the model.
            tools: The tools to use in the response.
            response_schema: The schema to use in the response.
            temperature: The temperature to use in the response.
            max_tokens: The maximum number of tokens to use in the response.
            top_p: The top p to use in the response.
        
        Returns:
            An OpenAgentResponse object.

        Example:
        ```python
        from openagentkit.tools import duckduckgo_search_tool
        from openagentkit.modules.openai import OpenAILLMService

        llm_service = OpenAILLMService(client, tools=[duckduckgo_search_tool])
        response = llm_service.model_generate(messages=[{"role": "user", "content": "What is TECHVIFY?"}])
        ```
        """

        temperature = kwargs.get("temperature", temperature)
        if temperature is None:
            temperature = self._temperature

        max_tokens = kwargs.get("max_tokens", max_tokens)
        if max_tokens is None:
            max_tokens = self._max_tokens

        top_p = kwargs.get("top_p", top_p)
        if top_p is None:
            top_p = self._top_p

        if tools is None:
            tools = self.tools
            
        #logger.info(f"Tools: {tools}")

        # Handle the client request
        response = self._handle_client_request(
            messages=messages, 
            tools=tools,
            response_schema=response_schema, 
        )
        
        if response.tool_calls:
            # Extract tool_calls arguments using the tool handler
            tool_calls = self._tool_handler.parse_tool_args(response)
            
            # Update the response with the parsed tool calls
            response.tool_calls = tool_calls
        
        return response

    def _handle_client_stream(self,
                              messages: List[Dict[str, str]],
                              tools: Optional[List[Dict[str, Any]]] = None,
                              response_schema: BaseModel = NOT_GIVEN,
                              temperature: Optional[float] = 0.3,
                              max_tokens: Optional[int] = None,
                              top_p: Optional[float] = None,
                              **kwargs) -> Generator[OpenAgentStreamingResponse, None, None]:
        """
        Handle the client stream.

        Args:
            messages: The messages to send to the model.
            tools: The tools to use in the response.
            response_schema: The schema to use in the response. **(not implemented yet)**
            temperature: The temperature to use in the response.
            max_tokens: The max tokens to use in the response.
            top_p: The top p to use in the response.

        Returns:
            An AsyncGenerator[OpenAgentStreamingResponse, None] object.
        """

        temperature = kwargs.get("temperature", temperature)
        if temperature is None:
            temperature = self._temperature

        max_tokens = kwargs.get("max_tokens", max_tokens)
        if max_tokens is None:
            max_tokens = self._max_tokens

        top_p = kwargs.get("top_p", top_p)
        if top_p is None:
            top_p = self._top_p
        
        if tools == NOT_GIVEN:
            tools = self._llm_service.tools
        
        
        if tools is None:
            tools = self.tools
        
        if response_schema is NOT_GIVEN:
            client_stream = self._client.chat.completions.create(
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

                        final_tool_calls[index].function.arguments += tool_call.function.arguments
            
            # After the stream is done, yield the final response with usage info if available
            if final_chunk and hasattr(final_chunk, 'usage'):
                yield OpenAgentStreamingResponse(
                    role="assistant",
                    content=final_content if final_content else None,
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
    
    def model_stream(self,
                     messages: List[Dict[str, str]],
                     tools: Optional[List[Dict[str, Any]]] = None,
                     response_schema: BaseModel = NOT_GIVEN,
                     temperature: Optional[float] = None,
                     max_tokens: Optional[int] = None,
                     top_p: Optional[float] = None,
                     **kwargs) -> Generator[OpenAgentStreamingResponse, None, None]:
        """
        Generate a response from the model.

        Args:
            messages: The messages to send to the model.
            tools: The tools to use in the response.
            response_schema: The schema to use in the response. **(not implemented yet)**
            temperature: The temperature to use in the response.
            max_tokens: The max tokens to use in the response.
            top_p: The top p to use in the response.

        Returns:
            An AsyncGenerator[OpenAgentStreamingResponse, None] object.
        """
        temperature = kwargs.get("temperature", temperature)
        if temperature is None:
            temperature = self._temperature

        max_tokens = kwargs.get("max_tokens", max_tokens)
        if max_tokens is None:
            max_tokens = self._max_tokens

        top_p = kwargs.get("top_p", top_p)
        if top_p is None:
            top_p = self._top_p

        if tools is None:
            tools = self.tools

        generator = self._handle_client_stream(
            messages=messages, 
            tools=tools,
            response_schema=response_schema,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        for chunk in generator:
            if chunk.tool_calls:
                # Extract tool_calls arguments using the tool handler
                tool_calls = self._tool_handler.parse_tool_args(chunk)
                
                # Update the chunk with the parsed tool calls
                chunk.tool_calls = tool_calls
            yield chunk

    def add_context(self, content: dict[str, str]):
        """
        Add context to the model.

        Args:
            content: The content to add to the context.

        Returns:
            The context history.
        """
        if not content:
            return self._context_history
        
        self._context_history.append(content)
        return self._context_history
        
    def extend_context(self, content: List[dict[str, str]]):
        """
        Extend the context of the model.

        Args:
            content: The content to extend the context with.

        Returns:
            The context history.
        """
        if not content:
            return self._context_history
        
        self._context_history.extend(content)
        return self._context_history
    
    def clear_context(self):
        """
        Clear the context of the model leaving only the system message.

        Returns:
            The cleared context history.
        """
        self._context_history = [
            {
                "role": "system",
                "content": self._system_message,
            }
        ]
        return self._context_history