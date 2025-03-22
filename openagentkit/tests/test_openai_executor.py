import unittest
from unittest.mock import MagicMock, patch
import os
import json
from datetime import datetime
from openai._types import NOT_GIVEN
from pydantic import BaseModel
from typing import Dict, List, Optional

from openagentkit.modules.openai.openai_executor import OpenAIExecutor
from openagentkit.modules.openai import OpenAILLMService
from openagentkit.models.responses import OpenAgentResponse


class TestResponseSchema(BaseModel):
    message: str
    confidence: float


class TestOpenAIExecutor(unittest.TestCase):
    def setUp(self):
        # Create mock OpenAI client
        self.mock_client = MagicMock()
        
        # Mock response for model_generate
        self.mock_response = MagicMock()
        self.mock_response.model_dump.return_value = {
            "content": "This is a test response",
            "role": "assistant"
        }
        
        # Create a mock for OpenAILLMService - Fix the patch target
        self.mock_llm_service_patcher = patch('openagentkit.modules.openai.openai_executor.OpenAILLMService')
        self.mock_llm_service_class = self.mock_llm_service_patcher.start()
        self.mock_llm_service = self.mock_llm_service_class.return_value
        self.mock_llm_service.model_generate.return_value = self.mock_response
        self.mock_llm_service.extend_context.return_value = [{"role": "user", "content": "Hello"}]
        self.mock_llm_service.add_context.return_value = [{"role": "user", "content": "Hello"}, 
                                                          {"role": "assistant", "content": "Hi there"}]
        self.mock_llm_service._context_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        self.mock_llm_service.tools = []
        
        # Create test instance with mocked dependencies
        self.executor = OpenAIExecutor(
            client=self.mock_client,
            model="gpt-4o-mini",
            system_message="Test system message",
            temperature=0.5
        )
        
        # We don't need to replace the LLM service with our mock anymore
        # since the patch will ensure it's already using our mock
        # self.executor._llm_service = self.mock_llm_service

    def tearDown(self):
        self.mock_llm_service_patcher.stop()

    def test_init(self):
        """Test that the executor initializes correctly"""
        self.assertEqual(self.executor._temperature, 0.5)
        self.assertEqual(self.executor._max_tokens, None)
        self.assertEqual(self.executor._top_p, None)
        self.assertEqual(self.executor._context_history, [])
        self.mock_llm_service_class.assert_called_once()

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
        
        result = self.executor.execute(messages)
        
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
        
        # Update mock response for schema
        schema_response = MagicMock()
        schema_response.model_dump.return_value = {
            "message": "Test message",
            "confidence": 0.95
        }
        self.mock_llm_service.model_generate.return_value = schema_response
        self.mock_llm_service._context_history[-1] = {
            "role": "assistant", 
            "content": {"message": "Test message", "confidence": 0.95}
        }
        
        result = self.executor.execute(messages, response_schema=schema)
        
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
        
        # Set up the mocks
        self.mock_llm_service.model_generate.side_effect = [tool_response, self.mock_response]
        self.mock_llm_service._handle_tool_call.return_value = tool_result
        
        # Execute with tool
        result = self.executor.execute(messages)
        
        # Verify tool handling
        self.mock_llm_service._handle_tool_call.assert_called_once_with("test_tool", param1="value1")
        self.assertEqual(self.mock_llm_service.model_generate.call_count, 2)
        
        # Verify result
        self.assertIsInstance(result, OpenAgentResponse)

    def test_delegation_methods(self):
        """Test that delegation methods correctly call LLM service"""
        messages = [{"role": "user", "content": "Hello"}]
        message = {"role": "assistant", "content": "Hi"}
        
        # Test extend_context
        self.executor.extend_context(messages)
        self.mock_llm_service.extend_context.assert_called_with(messages)
        
        # Test add_context
        self.executor.add_context(message)
        self.mock_llm_service.add_context.assert_called_with(message)
        
        # Test _handle_tool_call
        self.executor._handle_tool_call("test_tool", param1="value1")
        self.mock_llm_service._handle_tool_call.assert_called_with("test_tool", param1="value1")


if __name__ == "__main__":
    unittest.main() 