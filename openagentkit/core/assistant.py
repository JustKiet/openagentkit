import os
from typing import Union
from openagentkit.models.responses import OpenAgentResponse
from openagentkit.handlers import ContextHandler, OutputHandler

from openagentkit.interfaces import BaseExecutor

from dotenv import load_dotenv
from openai._types import NOT_GIVEN
from loguru import logger
from pydantic import BaseModel
import base64

load_dotenv()
    
class AIAssistant:
    def __init__(self,
                 context_handler: ContextHandler,
                 executor: BaseExecutor,
                 output_handler: OutputHandler
                ):
        self.context_handler = context_handler
        self.executor = executor
        self.output_handler = output_handler

    def set_system_message(self, system_message: str) -> str:
        return self.executor.define_system_message(
            system_message=system_message
        )
        
    def invoke(self, 
               message: str,
               speech: bool = False) -> Union[OpenAgentResponse, BaseModel]:
        # Check for context
        context = self.context_handler.get_context(message)
        
        llm_response = self.executor.execute(
            messages=[
                {
                    "role": "user",
                    "content": context,
                }
            ]
        )
        
        return self.output_handler.handle_output(
            output=llm_response,
            speech=speech
        )
