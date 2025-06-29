from openagentkit.core.interfaces import AsyncBaseContextStore
from typing import Any

class AsyncInMemoryContextStore(AsyncBaseContextStore):
    def __init__(self) -> None:
        self._context_store: dict[str, list[dict[str, Any]]] = {}

    async def get_system_message(self, context_id: str) -> str:
        """
        Get the system message for the given context ID.

        :param str context_id: The ID of the context.
        :return: The system message, or an empty string if not found.
        :rtype: str
        """
        context = self._context_store.get(context_id, [])
        
        for item in context:
            if item.get("role") == "system":
                return item.get("content", "")
    
        return ""
    
    async def update_system_message(self, context_id: str, system_message: str) -> None:
        self._context_store.setdefault(context_id, [])
        context = self._context_store[context_id]
        # Check if a system message already exists
        for item in context:
            if item.get("role") == "system":
                item["content"] = system_message
                return

    def init_context(self, context_id: str, system_message: str) -> list[dict[str, Any]]:
        """
        Initialize the context for the given context ID.

        :param str context_id: The ID of the context to initialize.
        :param str system_message: The system message to set for the context.
        :return: The initialized context, which includes the system message.
        :rtype: list[dict[str, Any]]
        """
        if context_id not in self._context_store:
            self._context_store[context_id] = []
            # Add a default system message if needed
            self._context_store[context_id].append({"role": "system", "content": system_message})
        return self._context_store[context_id]

    async def get_context(self, context_id: str) -> list[dict[str, Any]]:
        """
        Get the context for the given context ID.

        :param str context_id: The ID of the context.
        :return: The context as a list of message dictionaries.
        :rtype: list[dict[str, Any]]
        """
        return self._context_store.get(context_id, [])
        

    async def add_context(self, context_id: str, content: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Add context to the model.

        :param str context_id: The ID of the context to add content to.
        :param dict[str, Any] content: The content to add to the context.
        :return: The updated context history.
        :rtype: list[dict[str, Any]]
        """
        if context_id not in self._context_store:
            self._context_store[context_id] = []
        
        self._context_store[context_id].append(content)
        return self._context_store[context_id]

    async def extend_context(self, context_id: str, content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Extend the context of the model.

        :param str context_id: The ID of the context to extend.
        :param list[dict[str, Any]] content: The list of content to extend the context with.
        :return: The updated context history.
        :rtype: list[dict[str, Any]]
        """
        if context_id not in self._context_store:
            self._context_store[context_id] = []
        
        self._context_store[context_id].extend(content)
        return self._context_store[context_id]

    async def clear_context(self, context_id: str) -> list[dict[str, Any]]:
        """
        Clear the context of the model leaving only the system message.

        :param str context_id: The ID of the context to clear.
        :return: The updated context history.
        :rtype: list[dict[str, Any]]
        """
        if context_id in self._context_store:
            system_message = self.get_system_message(context_id)
            self._context_store[context_id] = [{"role": "system", "content": system_message}]
        
        return self._context_store[context_id]