from openagentkit.core.interfaces.base_context_store import BaseContextStore
from openagentkit.core.models.io.context_unit import ContextUnit
from openagentkit.core.exceptions import OperationNotAllowedError
from datetime import datetime
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

class InMemoryContextStore(BaseContextStore):
    def __init__(self) -> None:
        self._storage: dict[str, ContextUnit] = {}

    @property
    def storage(self) -> dict[str, ContextUnit]:
        """
        Get the context store.

        :return: The context store as a dictionary.
        :rtype: dict[str, ContextUnit]
        """
        return self._storage

    def get_system_message(self, thread_id: str) -> str:
        """
        Get the system message for the given thread ID.

        :param str thread_id: The ID of the context.
        :return: The system message, or an empty string if not found.
        :rtype: str
        """
        context = self._storage.get(thread_id, None)
        if context:
            for message in context.history:
                if message['role'] == 'system':
                    return message['content']
        # If no system message is found, return an empty string
        return ""

    def update_system_message(self, thread_id: str, agent_id: str, system_message: str) -> None:
        context = self._storage.get(thread_id, None)
        if context is None:
            context = self.init_context(thread_id, agent_id, system_message)
        else:
            if context.agent_id != agent_id:
                raise OperationNotAllowedError(f"Thread ID {thread_id} belongs to a different agent: {context.agent_id}, the provided agent ID is {agent_id}.")
            
            # Update the system message in the existing context
            for message in context.history:
                if message['role'] == 'system':
                    message['content'] = system_message
                    break
            else:
                # If no system message exists, add a new one at the beginning of the history
                context.history.insert(0, {"role": "system", "content": system_message})
        context.updated_at = int(datetime.now().timestamp())

    def init_context(
        self, 
        thread_id: str, 
        agent_id: str,
        system_message: str
    ) -> ContextUnit:
        """
        Initialize the context for the given thread ID.

        :param str thread_id: The ID of the context to initialize.
        :param str agent_id: The ID of the agent associated with the context.
        :param str system_message: The system message to set for the context.
        :return: The initialized context, which includes the system message.
        :rtype: ContextUnit
        """
        if thread_id not in self._storage:
            self._storage[thread_id] = ContextUnit(
                thread_id=thread_id,
                agent_id=agent_id,
                history=[{"role": "system", "content": system_message}],
                created_at=int(datetime.now().timestamp()),
                updated_at=int(datetime.now().timestamp())
            )
        else:
            raise OperationNotAllowedError(f"Context with thread ID {thread_id} already exists.")
        
        return self._storage[thread_id]

    def get_context(self, thread_id: str) -> Optional[ContextUnit]:
        """
        Get the context for the given thread ID.

        :param str thread_id: The ID of the context.
        :return: The context as a ContextUnit.
        :rtype: Optional[ContextUnit]
        """
        return self._storage.get(thread_id, None)

    def add_context(self, thread_id: str, agent_id: str, content: dict[str, Any]) -> ContextUnit:
        """
        Add context to the model.

        :param str thread_id: The ID of the context to add content to.
        :param str agent_id: The ID of the agent associated with the context.
        :param dict[str, Any] content: The content to add to the context.
        :return: The updated context history.
        :rtype: list[dict[str, Any]]
        """
        if thread_id not in self._storage:
            self._storage[thread_id] = ContextUnit(
                thread_id=thread_id,
                agent_id=agent_id,
                created_at=int(datetime.now().timestamp()),
                updated_at=int(datetime.now().timestamp())
            )

        if self._storage[thread_id].agent_id != agent_id:
            raise OperationNotAllowedError(f"Thread ID {thread_id} belongs to a different agent: {self._storage[thread_id].agent_id}, the provided agent ID is {agent_id}.")
        
        self._storage[thread_id].history.append(content)
        self._storage[thread_id].updated_at = int(datetime.now().timestamp())
        return self._storage[thread_id]

    def extend_context(self, thread_id: str, agent_id: str, content: list[dict[str, Any]]) -> ContextUnit:
        """
        Extend the context of the model.

        :param str thread_id: The ID of the context to extend.
        :param str agent_id: The ID of the agent associated with the context.
        :param list[dict[str, Any]] content: The list of content to extend the context with.
        :return: The updated context history.
        :rtype: ContextUnit
        """
        if thread_id not in self._storage:
            self._storage[thread_id] = ContextUnit(
                thread_id=thread_id,
                agent_id=agent_id,
                created_at=int(datetime.now().timestamp()),
                updated_at=int(datetime.now().timestamp())
            )
        
        if self._storage[thread_id].agent_id != agent_id:
            raise OperationNotAllowedError(f"Thread ID {thread_id} belongs to a different agent: {self._storage[thread_id].agent_id}, the provided agent ID is {agent_id}.")

        self._storage[thread_id].history.extend(content)
        self._storage[thread_id].updated_at = int(datetime.now().timestamp())
        return self._storage[thread_id]

    def clear_context(self, thread_id: str) -> Optional[ContextUnit]:
        """
        Clear the context of the model leaving only the system message.

        :param str thread_id: The ID of the context to clear.
        :return: The cleared context with only the system message.
        :rtype: Optional[ContextUnit]
        """
        if thread_id in self._storage:
            system_message = self.get_system_message(thread_id)
            self._storage[thread_id] = ContextUnit(
                thread_id=thread_id,
                agent_id=self._storage[thread_id].agent_id,
                history=[{"role": "system", "content": system_message}],
                created_at=int(datetime.now().timestamp()),
                updated_at=int(datetime.now().timestamp())
            )
        else:
            logger.warning(f"Attempted to clear context for non-existent thread ID: {thread_id}")
            return None
        return self._storage[thread_id]