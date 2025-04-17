from typing import Any, Callable, Dict, List, Optional, Union, Generator
import os
from loguru import logger
from openai._types import NOT_GIVEN
from openai import OpenAI
from openagentkit.core.interfaces.base_executor import BaseExecutor
from openagentkit.modules.openai import OpenAILLMService
from openagentkit.core.models.responses import OpenAgentResponse, OpenAgentStreamingResponse
from openagentkit.core.handlers.tool_handler import ToolHandler
from pydantic import BaseModel
import json
import datetime

class OpenAIExecutor(BaseExecutor):
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
        self._llm_service = OpenAILLMService(
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
        """
        Get the model name.

        Returns:
            The model name.
        """
        return self._llm_service._model

    @property
    def temperature(self) -> float:
        return self._llm_service.temperature

    @property
    def max_tokens(self) -> int:
        return self._llm_service.max_tokens
    
    @property
    def top_p(self) -> float:
        return self._llm_service.top_p
    
    def clone(self) -> 'OpenAIExecutor':
        """
        Clone the OpenAIExecutor object.

        Returns:
            A new OpenAIExecutor object with the same parameters.
        """
        return OpenAIExecutor(
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

    def execute(self, 
                messages: List[Dict[str, str]],
                tools: Optional[List[Dict[str, Any]]] = NOT_GIVEN,
                response_schema: Optional[BaseModel] = NOT_GIVEN,
                temperature: Optional[float] = None,
                max_tokens: Optional[int] = None,
                top_p: Optional[float] = None,
                **kwargs,
               ) -> Generator[OpenAgentResponse, None, None]:
        """
        Execute the OpenAI model and return an OpenAgentResponse object.

        Args:
            messages (List[Dict[str, str]]): The messages to send to the model.
            tools (Optional[List[Dict[str, Any]]]): The tools to use in the response.
            response_schema (Optional[BaseModel]): The schema to use in the response.
            temperature (Optional[float]): The temperature to use in the response.
            max_tokens (Optional[int]): The maximum number of tokens to use in the response.
            top_p (Optional[float]): The top p to use in the response.

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
        
        context = self._llm_service.extend_context(messages)
        
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
            )

            logger.info(f"Response Received: {response}")
            
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

                yield OpenAgentResponse(
                    role=response.role,
                    content=str(response.content) if not isinstance(response.content, (BaseModel, type(None))) else response.content,
                    tool_calls=response.tool_calls,
                    refusal=response.refusal,
                    usage=response.usage,
                )

                # Handle tool requests and get the final response with tool results
                tool_response = self._tool_handler.handle_tool_request(
                    response=response,
                )

                yield OpenAgentResponse(
                    role="tool",
                    tool_results=tool_response.tool_results,
                )

                logger.debug(f"Tool Messages in Execute: {tool_response.tool_messages}") if debug else None

                context = self._llm_service.extend_context(tool_response.tool_messages)

                logger.debug(f"Context: {context}") if debug else None
            else:
                stop = True

            if response.content is not None:
                # Add the final response to the context (chat history)
                self._llm_service.add_context(
                    {
                        "role": response.role,
                        "content": str(response.content),
                    }
                )

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
                

    def stream_execute(self, 
                      messages: List[Dict[str, str]],
                      tools: Optional[List[Dict[str, Any]]] = NOT_GIVEN,
                      response_schema: Optional[BaseModel] = NOT_GIVEN,
                      temperature: Optional[float] = 0.3,
                      max_tokens: Optional[int] = None,
                      top_p: Optional[float] = None,
                      **kwargs,
                      ) -> Generator[OpenAgentStreamingResponse, None, None]:
        """
        Stream execute the OpenAI model and return an OpenAgentStreamingResponse object.

        Args:
            messages (List[Dict[str, str]]): The messages to send to the model.
            tools (Optional[List[Dict[str, Any]]]): The tools to use in the response.
            response_schema (Optional[BaseModel]): The schema to use in the response.
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
            )
            
            for chunk in response_generator:
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
                    
                    context = self._llm_service.extend_context(tool_response.tool_messages)
                    
                    logger.debug(f"Context in Stream Execute: {context}") if debug else None

                elif chunk.finish_reason == "stop":
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