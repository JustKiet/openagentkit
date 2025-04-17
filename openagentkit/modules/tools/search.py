from langchain_community.tools import DuckDuckGoSearchResults
from openagentkit.core.models.tool_responses import DuckDuckGoResponse
from openagentkit.core.utils.tool_wrapper import tool
from typing import List, Annotated

search_engine = DuckDuckGoSearchResults(output_format="list")

@tool(
    description="Search for information using DuckDuckGo's search engine.",
    _notification=True,
)
def duckduckgo_search_tool(
    query: Annotated[str, "The search query to look up."],
) -> DuckDuckGoResponse:
    """Search for information using DuckDuckGo's search engine."""
    return DuckDuckGoResponse(
        query=query,
        response=search_engine.invoke(query)
    )
