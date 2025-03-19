import unittest
from unittest.mock import MagicMock, patch
import os
import json
from datetime import datetime
from openai._types import NOT_GIVEN
from pydantic import BaseModel
from typing import Dict, List, Optional

from openagentkit.modules.openai.openai_llm_service import OpenAILLMService
from openagentkit.handlers.tool_handler import ToolHandler
from openagentkit.models import OpenAgentResponse


class TestResponseSchema(BaseModel):
    message: str
    confidence: float


class TestOpenAILLMService(unittest.TestCase):
    def setUp(self):
        # Create mock OpenAI client and responses
        self.mock_client = MagicMock()
        
        # Mock for regular chat completion
        self.mock_chat_completion = MagicMock()
        self.mock_chat_message = MagicMock()
        self.mock_chat_message.model_dump.return_value = {
            "content": "This is a test response",
            "role": "assistant"
        }
        self.mock_chat_message.refusal = False
        self.mock_chat_completion.choices = [MagicMock(message=self.mock_chat_message)]
        self.mock_client.chat.completions.create.return_value = self.mock_chat_completion
        
        # Mock for beta chat completion with schema
        self.mock_beta_completion = MagicMock()
        self.mock_beta_message = MagicMock()
        self.mock_beta_message.parsed = {"message": "Test message", "confidence": 0.95}
        self.mock_beta_message.refusal = False
        self.mock_beta_completion.choices = [MagicMock(message=self.mock_beta_message)]
        self.mock_client.beta.chat.completions.parse.return_value = self.mock_beta_completion
        
        # Mock for ToolHandler
        self.mock_tool_handler_patcher = patch('openagentkit.modules.openai.openai_llm_service.ToolHandler')
        self.mock_tool_handler_class = self.mock_tool_handler_patcher.start()
        self.mock_tool_handler = MagicMock()
        self.mock_tool_handler_class.return_value = self.mock_tool_handler
        self.mock_tool_handler.tools = []
        self.mock_tool_handler.parse_tool_args.return_value = []
        self.mock_tool_handler._handle_tool_call.return_value = MagicMock(
            model_dump=lambda: {"result": "Tool executed successfully"}
        )
        
        # Create test instance
        self.llm_service = OpenAILLMService(
            client=self.mock_client,
            model="gpt-4o-mini",
            system_message="Test system message",
            temperature=0.5
        )
        
        # Replace the service's tool handler with our mock
        self.llm_service._tool_handler = self.mock_tool_handler
        
    def tearDown(self):
        self.mock_tool_handler_patcher.stop()
        
    def test_init(self):
        """Test that the service initializes correctly"""
        self.assertEqual(self.llm_service._model, "gpt-4o-mini")
        self.assertEqual(self.llm_service._system_message, "Test system message")
        self.assertEqual(self.llm_service._temperature, 0.5)
        self.assertEqual(self.llm_service._max_tokens, None)
        self.assertEqual(self.llm_service._top_p, None)
        self.assertEqual(len(self.llm_service._context_history), 1)
        self.assertEqual(self.llm_service._context_history[0]["role"], "system")
        self.assertEqual(self.llm_service._context_history[0]["content"], "Test system message")
        
    def test_model_generate_basic(self):
        """Test basic model generation without schema"""
        messages = [{"role": "user", "content": "Hello"}]
        
        result = self.llm_service.model_generate(messages)
        
        # Verify client was called correctly
        self.mock_client.chat.completions.create.assert_called_once()
        call_args = self.mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_args["model"], "gpt-4o-mini")
        self.assertEqual(call_args["messages"], messages)
        self.assertEqual(call_args["temperature"], 0.5)
        
        # Verify tool args were parsed
        self.mock_tool_handler.parse_tool_args.assert_called_once()
        
        # Verify result
        self.assertIsInstance(result, OpenAgentResponse)
        self.assertEqual(result.content, "This is a test response")
        self.assertEqual(result.role, "assistant")
        
    def test_model_generate_with_schema(self):
        """Test model generation with response schema"""
        messages = [{"role": "user", "content": "Hello"}]
        schema = TestResponseSchema
        
        result = self.llm_service.model_generate(messages, response_schema=schema)
        
        # Verify beta client was called correctly
        self.mock_client.beta.chat.completions.parse.assert_called_once()
        call_args = self.mock_client.beta.chat.completions.parse.call_args[1]
        self.assertEqual(call_args["model"], "gpt-4o-mini")
        self.assertEqual(call_args["messages"], messages)
        self.assertEqual(call_args["temperature"], 0.5)
        self.assertEqual(call_args["response_format"], schema)
        
        # Verify result
        self.assertEqual(result, {"message": "Test message", "confidence": 0.95})
        
    def test_model_generate_with_refusal(self):
        """Test model generation with refusal"""
        messages = [{"role": "user", "content": "Hello"}]
        
        # Set up refusal response
        self.mock_chat_message.refusal = True
        
        result = self.llm_service.model_generate(messages)
        
        # Verify result is OpenAgentResponse
        self.assertIsInstance(result, OpenAgentResponse)
        
    def test_context_management(self):
        """Test context management methods"""
        # Test add_context
        message = {"role": "user", "content": "Hello"}
        result = self.llm_service.add_context(message)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1], message)
        
        # Test extend_context
        messages = [
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"}
        ]
        result = self.llm_service.extend_context(messages)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[2], messages[0])
        self.assertEqual(result[3], messages[1])
        
    def test_tool_handling(self):
        """Test tool handling delegation"""
        result = self.llm_service._handle_tool_call("test_tool", param1="value1")
        
        # Verify tool handler was called correctly
        self.mock_tool_handler._handle_tool_call.assert_called_once_with("test_tool", param1="value1")
        
        # Verify result
        self.assertEqual(result.model_dump(), {"result": "Tool executed successfully"})


if __name__ == "__main__":
    unittest.main() 