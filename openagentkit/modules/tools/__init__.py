"""
A module for pre-built tools.

## Tools:
    - `get_weather_tool`: A tool to get the weather.
    - `duckduckgo_search_tool`: A tool to search the web using DuckDuckGo.
"""

from .get_weather import get_weather_tool
from .search import duckduckgo_search_tool

__all__ = [
    'get_weather_tool',
    'duckduckgo_search_tool',
    'authenticate_tool',
    'save_chat_messages',
    'save_single_message',
    'register_user_tool'
]