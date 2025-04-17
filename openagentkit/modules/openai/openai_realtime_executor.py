from openagentkit.core.interfaces import AsyncBaseExecutor
from typing import Optional, List, Dict, Any, AsyncGenerator
from openagentkit.core.models.responses import OpenAgentResponse, OpenAgentStreamingResponse
import websockets
from openai import AsyncOpenAI, AsyncAzureOpenAI

class OpenAIRealtimeExecutor(AsyncBaseExecutor):
    def __init__(self, client: AsyncOpenAI):
        self.client = client

    def clone(self) -> 'OpenAIRealtimeExecutor':
        return OpenAIRealtimeExecutor(self.client)

    async def define_system_message(self, system_message: Optional[str]) -> str:
        return system_message

    async def execute(self, 
                      messages: List[Dict[str, str]], 
                      tools: Optional[List[Dict[str, Any]]], 
                      temperature: Optional[float] = None, 
                      max_tokens: Optional[int] = None, 
                      top_p: Optional[float] = None
                      ) -> AsyncGenerator[OpenAgentResponse, None]:
        pass

    async def stream_execute(self, 
                             messages: List[Dict[str, str]], 
                             tools: Optional[List[Dict[str, Any]]], 
                             temperature: Optional[float] = None, 
                             max_tokens: Optional[int] = None, 
                             top_p: Optional[float] = None
                             ) -> AsyncGenerator[OpenAgentStreamingResponse, None]:
        pass    

    def get_history(self) -> List[Dict[str, Any]]:
        return []

