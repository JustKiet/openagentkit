from openagentkit.modules.openai import AsyncOpenAIAgent
from openagentkit.core.context import InMemoryContextStore
import openai
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel
import os

load_dotenv()

class ResponseSchema(BaseModel):
    content: str

context_store = InMemoryContextStore()

async def main():
    client = openai.AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    agent = AsyncOpenAIAgent(
        client=client,
        system_message="You are a helpful assistant.",
        tools=[],
        context_store=context_store,
    )

    # Access the thread_id property
    print(f"Thread ID: {agent.thread_id}")

    async for event in agent.execute(
        messages=[
            {
                "role": "user",
                "content": "Hi, my name is John."
            }
        ]
    ):
        if event.content:
            print(f"Response: {event.content}")

    async for event in agent.execute(
        messages=[
            {
                "role": "user",
                "content": "What is my name?"
            }
        ]
    ):
        if event.content:
            print(f"Response: {event.content}")

    async for event in agent.execute(
        messages=[
            {
                "role": "user",
                "content": "What is my name?"
            }
        ],
        thread_id="new_context"
    ):
        if event.content:
            print(f"Response: {event.content}")

    async for event in agent.execute(
        messages=[
            {
                "role": "user",
                "content": "Okay lovely, can you refer me to my name at the end of your sentence always?"
            }
        ],
        response_schema=ResponseSchema
    ):
        if event.content:
            print(f"Response: {event.content}")
            print(f"Response content type: {type(event.content)}")
            assert isinstance(event.content, ResponseSchema), f"Response should be of type ResponseSchema, got {type(event.content)}"

if __name__ == "__main__":
    asyncio.run(main())
    print(context_store.storage)