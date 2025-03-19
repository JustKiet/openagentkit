import unittest
from unittest.mock import MagicMock, patch
import os
import json
import asyncio
from datetime import datetime
from openai._types import NOT_GIVEN
from pydantic import BaseModel
from typing import Dict, List, Optional

from openagentkit.modules.openai.async_openai_executor import AsyncOpenAIExecutor
from openagentkit.modules.openai.async_openai_llm_service import AsyncOpenAILLMService
from openagentkit.models import OpenAgentResponse


class TestResponseSchema(BaseModel):
    message: str
    confidence: float


class TestAsyncOpenAIExecutor(unittest.TestCase):
    def setUp(self):
        # Set up event loop for testing
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Create mock AsyncOpenAI client
        self.mock_client = MagicMock()
        
        # Mock response for model_generate
        self.mock_response = MagicMock()
        self.mock_response.model_dump.return_value = {
            "content": "This is a test response",
            "role": "assistant"
        }
        
        # Create a mock for AsyncOpenAILLMService
        self.mock_llm_service_patcher = patch('openagentkit.modules.openai.async_openai_executor.AsyncOpenAILLMService')
        self.mock_llm_service_class = self.mock_llm_service_patcher.start()
        self.mock_llm_service = MagicMock()
        self.mock_llm_service_class.return_value = self.mock_llm_service
        
        # Set up async method mocks with proper futures
        model_generate_future = self.loop.create_future()
        model_generate_future.set_result(self.mock_response)
        self.mock_llm_service.model_generate = MagicMock(return_value=model_generate_future)
        
        extend_context_future = self.loop.create_future()
        extend_context_future.set_result([{"role": "user", "content": "Hello"}])
        self.mock_llm_service.extend_context = MagicMock(return_value=extend_context_future)
        
        add_context_future = self.loop.create_future()
        add_context_future.set_result([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ])
        self.mock_llm_service.add_context = MagicMock(return_value=add_context_future)
        
        self.mock_llm_service._context_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        self.mock_llm_service.tools = []
        
        # Create test instance with mocked dependencies
        self.executor = AsyncOpenAIExecutor(
            client=self.mock_client,
            model="gpt-4o-mini",
            system_message="Test system message",
            temperature=0.5
        )
        
        # Replace the executor's LLM service with our mock
        self.executor._llm_service = self.mock_llm_service

    def tearDown(self):
        self.mock_llm_service_patcher.stop()
        self.loop.close()

    def test_init(self):
        """Test that the executor initializes correctly"""
        self.assertEqual(self.executor._temperature, 0.5)
        self.assertEqual(self.executor._max_tokens, None)
        self.assertEqual(self.executor._top_p, None)

    def test_define_system_message(self):
        """Test that system message is defined correctly"""
        # Test with provided message
        message = self.executor.define_system_message("Custom message")
        self.assertIn("Custom message", message)
        self.assertIn(datetime.now().strftime("%Y-%m-%d"), message)
        
        # Test with default message
        message = self.executor.define_system_message()
        self.assertIn("You are an helpful assistant", message)
        self.assertIn(datetime.now().strftime("%Y-%m-%d"), message)

    def test_execute_basic(self):
        """Test basic execution without tools or schema"""
        messages = [{"role": "user", "content": "Hello"}]
        
        # Run the async method in the event loop
        result = self.loop.run_until_complete(self.executor.execute(messages))
        
        # Verify interactions with LLM service
        self.mock_llm_service.extend_context.assert_called_once_with(messages)
        self.mock_llm_service.model_generate.assert_called_once()
        self.mock_llm_service.add_context.assert_called_once()
        
        # Verify result
        self.assertIsInstance(result, OpenAgentResponse)

    def test_execute_with_schema(self):
        """Test execution with response schema"""
        messages = [{"role": "user", "content": "Hello"}]
        schema = TestResponseSchema
        
        # Create a mock response that has model_dump method
        schema_response = MagicMock()
        schema_response.model_dump.return_value = {
            "message": "Test message", 
            "confidence": 0.95
        }
        
        # Create a future for the response
        schema_future = self.loop.create_future()
        schema_future.set_result(schema_response)
        
        # Create a new mock for this test
        schema_mock = MagicMock(return_value=schema_future)
        
        # Patch the model_generate method for this test only
        with patch.object(self.mock_llm_service, 'model_generate', schema_mock):
            # Set up context history for the response
            self.mock_llm_service._context_history = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": {"message": "Test message", "confidence": 0.95}}
            ]
            
            # Run the async method in the event loop
            result = self.loop.run_until_complete(self.executor.execute(messages, response_schema=schema))
            
            # Verify model_generate was called
            schema_mock.assert_called_once()
        
        # Verify result
        self.assertIsInstance(result, TestResponseSchema)
        self.assertEqual(result.message, "Test message")
        self.assertEqual(result.confidence, 0.95)

    def test_execute_with_tools(self):
        """Test execution with tool calls"""
        messages = [{"role": "user", "content": "Use a tool"}]
        
        # Mock tool call response
        tool_response = MagicMock()
        tool_response.model_dump.return_value = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": {
                        "name": "test_tool",
                        "arguments": "{'param1': 'value1'}"
                    }
                }
            ]
        }
        
        # Mock tool result
        tool_result = MagicMock()
        tool_result.model_dump.return_value = {"result": "Tool executed successfully"}
        
        # Create futures for the responses
        tool_response_future = self.loop.create_future()
        tool_response_future.set_result(tool_response)
        
        normal_response_future = self.loop.create_future()
        normal_response_future.set_result(self.mock_response)
        
        # Create a mock that returns different values on each call
        tool_mock = MagicMock(side_effect=[tool_response_future, normal_response_future])
        
        # Set up the tool call mock
        tool_call_future = self.loop.create_future()
        tool_call_future.set_result(tool_result)
        tool_call_mock = MagicMock(return_value=tool_call_future)
        
        # Patch both methods for this test
        with patch.object(self.mock_llm_service, 'model_generate', tool_mock), \
             patch.object(self.mock_llm_service, '_handle_tool_call', tool_call_mock):
            
            # Run the async method in the event loop
            result = self.loop.run_until_complete(self.executor.execute(messages))
            
            # Verify tool handling
            tool_call_mock.assert_called_once()
            self.assertEqual(tool_mock.call_count, 2)
        
        # Verify result
        self.assertIsInstance(result, OpenAgentResponse)


if __name__ == "__main__":
    unittest.main() 