from abc import ABC, abstractmethod
from typing import Any

class BaseContextStore(ABC):
    @abstractmethod
    def get_system_message(self, context_id: str) -> str:
        """
        Get the system message for the given context ID.

        :param str context_id: The ID of the context.
        :return: The system message, or an empty string if not found.
        :rtype: str
        """
        pass
    
    @abstractmethod
    def update_system_message(self, context_id: str, system_message: str) -> None:
        """
        Update the system message for the given context ID.

        :param str context_id: The ID of the context to update.
        :param str system_message: The new system message to set.
        """
        pass

    @abstractmethod
    def init_context(self, context_id: str, system_message: str) -> list[dict[str, Any]]:
        """
        Initialize the context for the given context ID.

        :param str context_id: The ID of the context to initialize.
        :param str system_message: The system message to set for the context.
        :return: The initialized context, which includes the system message.
        :rtype: list[dict[str, Any]]
        """
        pass
    
    @abstractmethod
    def get_context(self, context_id: str) -> list[dict[str, Any]]:
        """
        An abstract method to get the history of the conversation.

        :param str context_id: The ID of the context to retrieve.
        :return: The context history.
        :rtype: list[dict[str, Any]]
        """
        pass
    
    @abstractmethod
    def add_context(self, context_id: str, content: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Add context to the model.

        :param str context_id: The ID of the context to add content to.
        :param dict[str, Any] content: The content to add to the context.
        :return: The updated context history.
        :rtype: list[dict[str, Any]]
        """
        pass
    
    @abstractmethod
    def extend_context(self, context_id: str, content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Extend the context of the model.

        :param str context_id: The ID of the context to extend.
        :param list[dict[str, Any]] content: The list of content to extend the context with.
        :return: The updated context history.
        :rtype: list[dict[str, Any]]
        """
        pass
    
    @abstractmethod
    def clear_context(self, context_id: str) -> list[dict[str, Any]]:
        """
        Clear the context of the model leaving only the system message.

        :param str context_id: The ID of the context to clear.
        :return: The updated context history.
        :rtype: list[dict[str, Any]]
        """
        pass