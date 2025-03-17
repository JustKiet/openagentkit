from openagentkit.tools.search import search
from openagentkit.tools.get_weather import get_weather
from openagentkit.modules.openai import OpenAISpeechService
from openagentkit.modules.smallestai.lightning_speech_service import LightningSpeechService
from openagentkit.handlers import ContextHandler, OutputHandler, EventHandler
from openagentkit.modules.common import Executor
from openagentkit.core import AIAssistant
from openai import AzureOpenAI
import pprint
import time
from dotenv import load_dotenv
import os

load_dotenv()

client = AzureOpenAI()

speech_client = LightningSpeechService()

context_handler = ContextHandler()

event_handler = EventHandler()

executor = Executor(
    client=client,
    tools=[search, get_weather]
)

output_handler = OutputHandler(
    speech_client=speech_client,
    event_handler=event_handler
)

assistant = AIAssistant(
    context_handler=context_handler,
    executor=executor,
    output_handler=output_handler
)

start_time = time.time()

response = assistant.invoke(
    message="Research on the movie 'A Minecraft Movie'."
)

end_time = time.time()

print(f"Time taken: {end_time - start_time}")