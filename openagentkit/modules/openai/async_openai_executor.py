from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union
import os
from loguru import logger
from openai._types import NOT_GIVEN
from openai import AsyncOpenAI
from openagentkit.interfaces.async_base_executor import AsyncBaseExecutor
from openagentkit.modules.openai.async_openai_llm_service import AsyncOpenAILLMService
from openagentkit.models.responses import OpenAgentResponse, OpenAIStreamingResponse
from openagentkit.models.tool_responses import ToolResultResponse
from pydantic import BaseModel
import json
import datetime

class AsyncOpenAIExecutor(AsyncBaseExecutor):
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
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p

    def define_system_message(self, message: Optional[str] = None) -> str:
        system_message = message if message is not None else """
            System Message: You are an helpful assistant, try to assist the user in everything.\n
            """
        system_message += f"""
        Current date and time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n
        
        """
        return system_message
    
    async def _handle_tool_request(self, response: OpenAgentResponse) -> tuple[list, list]:
        """
        Handle tool requests and get the final response with tool results
        """
        assert type(response) == OpenAgentResponse or type(response) == OpenAIStreamingResponse, "Response must be an OpenAgentResponse or OpenAIStreamingResponse object"
        tool_results = []
        tool_messages = []
        
        # Check if the response contains tool calls
        if response.tool_calls is None:
            logger.debug("No tool calls found in the response. Skipping tool call handling.")
            return tool_results, tool_messages

        # Handle tool calls 
        for tool_call in response.tool_calls:
            tool_call_id = tool_call.get("id")
            tool_name = tool_call.get("function").get("name")
            tool_args = eval(tool_call.get("function").get("arguments"))
            # Save notification value and remove _notification key from tool args if present
            notification = tool_args.get("_notification", None)
            tool_args.pop("_notification", None)
            tool_result = await self._llm_service._handle_tool_call(tool_name, **tool_args)
            
            # Store tool call and result
            tool_results.append({
                "tool_call": tool_call,
                "result": tool_result,
            })
            
            logger.info(f"Tool Result: {tool_result}")
            
            # Convert tool result to string if it's not already a string
            tool_result_str = str(tool_result)
            
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": tool_result_str,  # Use string representation
            }

            tool_messages.append(tool_message)
        
        return tool_results, tool_messages, notification

    async def execute(self, 
                messages: List[Dict[str, str]],
                tools: Optional[List[Dict[str, Any]]] = NOT_GIVEN,
                response_schema: Optional[BaseModel] = NOT_GIVEN,
                verbose: bool = False,
                **kwargs,
               ) -> Union[OpenAgentResponse, BaseModel, AsyncGenerator[Dict[str, Any], None]]:
        kwargs.get("temperature", self._temperature)
        kwargs.get("max_tokens", self._max_tokens)
        kwargs.get("top_p", self._top_p)
        
        if tools == NOT_GIVEN:
            tools = self._llm_service.tools
        
        context = await self._llm_service.extend_context(messages)
        
        logger.debug(f"Context: {context}")
        
        # Take user intial request along with the chat history -> response
        response = await self._llm_service.model_generate(
            messages=context, 
            tools=tools, 
            response_schema=response_schema
        )
        
        # Add the response to the context (chat history)
        context = await self._llm_service.add_context(response.model_dump())
        
        logger.info(f"Response Received: {response}")

        tool_results = []
        
        if response.tool_calls:
            # Handle tool requests abd get the final response with tool results
            tool_results, tool_messages, notification = await self._handle_tool_request(
                response=response,
            )

            logger.debug(f"Tool Messages in Execute: {tool_messages}")

            context = await self._llm_service.extend_context(tool_messages)

            logger.debug(f"Context: {context}")

            # Generate the final response with the tool results
            response = await self._llm_service.model_generate(
                messages=context,
                tools=tools, 
                response_schema=response_schema
            )

        # Add the final response to the context (chat history)
        await self._llm_service.add_context(
            {
                "role": response.role,
                "content": response.content,
            }
        )
        
        logger.debug(f"Final Response: {response}")
        
        # If there was a response schema, return the response schema
        if response_schema:
            return response_schema(
                **response.get("content"),
            )
        
        # If there is no response, return an error
        if not response:
            logger.error("No response from the model")
            return OpenAgentResponse(
                role="assistant",
                content="",
                tool_results=tool_results,
                refusal="No response from the model",
                audio=None,
            )
        
        # Create a response that includes the assistant's content, tool calls, and results
        response_data = {
            "role": response.role,
            "content": response.content,
            "tool_results": tool_results,
        }
        
        # Add tool calls if they exist
        if "tool_calls" in response:
            response_data["tool_calls"] = response.get("tool_calls")
        
        return OpenAgentResponse(**response_data)

    async def stream_execute(self, 
                             messages: List[Dict[str, str]],
                             tools: Optional[List[Dict[str, Any]]] = NOT_GIVEN,
                             response_schema: Optional[BaseModel] = NOT_GIVEN,
                             **kwargs,
                             ) -> AsyncGenerator[OpenAIStreamingResponse, None]:
        kwargs.get("temperature", self._temperature)
        kwargs.get("max_tokens", self._max_tokens)
        kwargs.get("top_p", self._top_p)
        
        if tools == NOT_GIVEN:
            tools = self._llm_service.tools

        context = await self._llm_service.extend_context(messages)

        logger.debug(f"Context: {context}")

        response_generator = self._llm_service.model_stream(
            messages=context,
            tools=tools,
            response_schema=response_schema,
        )

        final_response_generator = None
        
        async for chunk in response_generator:
            if chunk.finish_reason == "tool_calls":
                # Add the llm tool call request to the context
                context = await self._llm_service.add_context(chunk.model_dump())

                logger.debug(f"Context: {context}")

                notification = chunk.tool_calls[0].get("function")
                tool_notification = None

                if notification.get("arguments"):
                    if type(notification.get("arguments")) == str:
                        args = json.loads(notification.get("arguments"))
                    else:
                        args = notification.get("arguments")

                    if args.get("_notification"):
                        tool_notification = args.get("_notification", None)

                    if notification:
                        logger.info(f"Tool Notification: {tool_notification}")
                        yield OpenAIStreamingResponse(
                            role="assistant",
                            content="",
                            tool_notification=tool_notification,
                        )

                # Handle the tool call request and get the final response with tool results
                _, tool_messages, _ = await self._handle_tool_request(
                    response=chunk,
                ) 

                context = await self._llm_service.extend_context(tool_messages)

                logger.debug(f"Tool Messages in Execute: {tool_messages}")
                
                logger.debug(f"Context in Stream Execute: {context}")
                
                final_response_generator = self._llm_service.model_stream(
                    messages=context,
                    tools=tools,
                    response_schema=response_schema,
                )

                async for chunk in final_response_generator:
                    if chunk.content:
                        context = await self._llm_service.add_context({"role": "assistant", "content": chunk.content})
                        logger.info(f"Context: {context}")
                    yield chunk

            elif chunk.finish_reason == "stop":
                if chunk.content:
                    context = await self._llm_service.add_context({"role": "assistant", "content": chunk.content})
                    logger.info(f"Context: {context}")
                    yield chunk
            else:
                yield chunk
