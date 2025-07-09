from openagentkit.core.models.io.context_unit import ContextUnit
from abc import ABC, abstractmethod
from typing import Any, Optional

class BaseContextStore(ABC):
    @abstractmethod
    def get_system_message(self, thread_id: str) -> str:
        """
        Get the system message for the given thread ID.

        :param str thread_id: The ID of the context.
        :return: The system message, or an empty string if not found.
        :rtype: str
        """
        pass
    
    @abstractmethod
    def update_system_message(self, thread_id: str, agent_id: str, system_message: str) -> None:
        """
        Update the system message for the given thread ID.

        :param str thread_id: The ID of the context to update.
        :param str agent_id: The ID of the agent associated with the context.
        :param str system_message: The new system message to set.
        """
        pass

    @abstractmethod
    def init_context(self, thread_id: str, agent_id: str, system_message: str) -> ContextUnit:
        """
        Initialize the context for the given thread ID. If the context already exists, it will return the existing context (Only if the agent_id matches).

        :param str thread_id: The ID of the context to initialize.
        :param str agent_id: The ID of the agent associated with the context.
        :param str system_message: The system message to set for the context.
        :return: The initialized context, which includes the system message.
        :rtype: ContextUnit
        """
        pass
    
    @abstractmethod
    def get_context(self, thread_id: str) -> Optional[ContextUnit]:
        """
        An abstract method to get the history of the conversation.

        :param str thread_id: The ID of the context to retrieve.
        :return: The context history.
        :rtype: Optional[ContextUnit]
        """
        pass

    @abstractmethod
    def get_agent_context(self, agent_id: str) -> dict[str, ContextUnit]:
        """
        Get all contexts associated with a specific agent ID.

        :param str agent_id: The ID of the agent to retrieve contexts for.
        :return: A dictionary mapping thread IDs to their respective ContextUnit objects.
        :rtype: dict[str, ContextUnit]
        """
        pass
    
    @abstractmethod
    def add_context(
        self, 
        thread_id: str, 
        agent_id: str, 
        content: dict[str, Any], 
        system_message: Optional[str] = None
    ) -> ContextUnit:
        """
        Add context to the model.

        :param str thread_id: The ID of the context to add content to.
        :param str agent_id: The ID of the agent associated with the context.
        :param dict[str, Any] content: The content to add to the context.
        :param Optional[str] system_message: An optional system message to set for the context in case thread_id does not exist.
        :return: The updated context history.
        :rtype: ContextUnit
        """
        pass
    
    @abstractmethod
    def extend_context(
        self, 
        thread_id: str, 
        agent_id: str, 
        content: list[dict[str, Any]], 
        system_message: Optional[str] = None
    ) -> ContextUnit:
        """
        Extend the context of the model.

        :param str thread_id: The ID of the context to extend.
        :param str agent_id: The ID of the agent associated with the context.
        :param list[dict[str, Any]] content: The list of content to extend the context with.
        :param Optional[str] system_message: An optional system message to set for the context in case thread_id does not exist.
        :return: The updated context history.
        :rtype: ContextUnit
        """
        pass
    
    @abstractmethod
    def clear_context(self, thread_id: str) -> Optional[ContextUnit]:
        """
        Clear the context of the model leaving only the system message.

        :param str thread_id: The ID of the context to clear.
        :return: The updated context history.
        :rtype: Optional[ContextUnit]
        """
        pass