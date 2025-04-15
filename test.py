from openagentkit.modules.openai import AsyncOpenAIExecutor
from openagentkit.utils.tool_wrapper import tool
from pydantic import BaseModel
import openai
import os
from typing import Annotated
import asyncio
# Define a tool
@tool(
    description="Get the weather of a city",
)
def get_weather(city: Annotated[str, "The city to get the weather of"]) -> str:
    """Get the weather of a city"""

    class WeatherResponse(BaseModel):
        city: str
        weather: str
        temperature: float
        feels_like: float
        humidity: float

    return WeatherResponse(
        city=city,
        weather="sunny",
        temperature=20,
        feels_like=22,
        humidity=0.5,
    )

# Initialize OpenAI client
client = openai.AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

async def main():
    # Initialize LLM service
    executor = AsyncOpenAIExecutor(
        client=client,
        model="gpt-4o-mini",
        system_message="""
        You are a helpful assistant that can answer questions and help with tasks.
        You are also able to use tools to get information.
        """,
        tools=[get_weather],
        temperature=0.5,
        max_tokens=100,
        top_p=1.0,
    )

    generator = executor.execute(
        messages=[
            {"role": "user", "content": "What's the weather like in New York?"}
        ]
    )

    async for response in generator:
        print(response.content)

if __name__ == "__main__":
    asyncio.run(main())