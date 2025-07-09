from openagentkit.core.interfaces.base_context_store import BaseContextStore
from openagentkit.core.models.io.context_unit import ContextUnit
from openagentkit.core.exceptions import OperationNotAllowedError
from datetime import datetime
from typing import Any, Optional
import logging
import threading

logger = logging.getLogger(__name__)

class InMemoryContextStore(BaseContextStore):
    def __init__(self) -> None:
        self._storage: dict[str, ContextUnit] = {}
        self._thread_locks: dict[str, threading.Lock] = {} # New: dictionary to hold locks per thread_id
        self._main_lock = threading.Lock() # New: Lock to protect _storage and _thread_locks dictionary itself

    def _get_or_create_thread_lock(self, thread_id: str) -> threading.Lock:
        """Helper to get or create a lock for a specific thread_id."""
        with self._main_lock: # Protect access to _thread_locks dictionary
            if thread_id not in self._thread_locks:
                self._thread_locks[thread_id] = threading.Lock()
            return self._thread_locks[thread_id]

    def get_system_message(self, thread_id: str) -> str:
        thread_lock = self._get_or_create_thread_lock(thread_id)
        with thread_lock: # Acquire lock specific to this thread_id
            context = self._storage.get(thread_id, None)
            if context:
                for message in context.history:
                    if message['role'] == 'system':
                        return message['content']
            return ""

    def update_system_message(self, thread_id: str, agent_id: str, system_message: str) -> None:
        thread_lock = self._get_or_create_thread_lock(thread_id)
        with thread_lock: # Acquire lock specific to this thread_id
            context = self._storage.get(thread_id, None)
            if context is None:
                # If context doesn't exist, initialize it (this method will also use the same thread_lock)
                self.init_context(thread_id, agent_id, system_message)
            else:
                # IMPORTANT: Perform agent_id check *inside* the locked section
                if context.agent_id != agent_id:
                    raise OperationNotAllowedError(f"Thread ID {thread_id} belongs to a different agent: {context.agent_id}, the provided agent ID is {agent_id}.")
                
                for message in context.history:
                    if message['role'] == 'system':
                        message['content'] = system_message
                        break
                else: # Only runs if loop completes without `break`
                    # If no system message exists, add a new one at the beginning of the history
                    context.history.insert(0, {"role": "system", "content": system_message})
                context.updated_at = int(datetime.now().timestamp())

    def init_context(
        self, 
        thread_id: str, 
        agent_id: str,
        system_message: str
    ) -> ContextUnit:
        thread_lock = self._get_or_create_thread_lock(thread_id)
        with thread_lock: # Acquire lock specific to this thread_id
            if thread_id in self._storage: # Check again under the lock
                if self._storage[thread_id].agent_id != agent_id:
                    raise OperationNotAllowedError(f"Context with thread ID {thread_id} already exists.")
                
                if system_message != self._storage[thread_id].system_message:
                    for message in self._storage[thread_id].history:
                        if message['role'] == 'system':
                            message['content'] = system_message
                            break
                    else:
                        self._storage[thread_id].history.insert(0, {"role": "system", "content": system_message})
                    self._storage[thread_id].updated_at = int(datetime.now().timestamp())
                
                return self._storage[thread_id] # Return existing context if it matches agent_id
            
            self._storage[thread_id] = ContextUnit(
                thread_id=thread_id,
                agent_id=agent_id,
                history=[{"role": "system", "content": system_message}],
                created_at=int(datetime.now().timestamp()),
                updated_at=int(datetime.now().timestamp())
            )
            return self._storage[thread_id]

    def get_context(self, thread_id: str) -> Optional[ContextUnit]:
        thread_lock = self._get_or_create_thread_lock(thread_id)
        with thread_lock: # Acquire lock specific to this thread_id
            return self._storage.get(thread_id, None)
        
    def get_agent_context(self, agent_id: str) -> dict[str, ContextUnit]:
        with self._main_lock:
            agent_contexts = {
                thread_id: context 
                for thread_id, context in self._storage.items() 
                if context.agent_id == agent_id
            }
        return agent_contexts

    def add_context(
        self, 
        thread_id: str, 
        agent_id: str, 
        content: dict[str, Any],
        system_message: Optional[str] = None
    ) -> ContextUnit:
        thread_lock = self._get_or_create_thread_lock(thread_id)
        with thread_lock: # Acquire lock specific to this thread_id
            if thread_id not in self._storage:
                if not system_message:
                    raise ValueError("System message must be provided when initializing a new context.")
                
                self._storage[thread_id] = ContextUnit(
                    thread_id=thread_id,
                    agent_id=agent_id, # This assigns the agent_id to a newly created context
                    created_at=int(datetime.now().timestamp()),
                    updated_at=int(datetime.now().timestamp()),
                    history=[{"role": "system", "content": system_message}]
                )
            
            # IMPORTANT: Perform agent_id check *inside* the locked section
            if self._storage[thread_id].agent_id != agent_id:
                raise OperationNotAllowedError(f"Thread ID {thread_id} belongs to a different agent: {self._storage[thread_id].agent_id}, the provided agent ID is {agent_id}.")
            
            self._storage[thread_id].history.append(content)
            self._storage[thread_id].updated_at = int(datetime.now().timestamp())
            return self._storage[thread_id]

    def extend_context(
        self, 
        thread_id: str, 
        agent_id: str, 
        content: list[dict[str, Any]],
        system_message: Optional[str] = None
    ) -> ContextUnit:
        thread_lock = self._get_or_create_thread_lock(thread_id)
        with thread_lock: # Acquire lock specific to this thread_id
            if thread_id not in self._storage:
                if not system_message:
                    raise ValueError("System message must be provided when initializing a new context.")
                
                self._storage[thread_id] = ContextUnit(
                    thread_id=thread_id,
                    agent_id=agent_id, # This assigns the agent_id to a newly created context
                    created_at=int(datetime.now().timestamp()),
                    updated_at=int(datetime.now().timestamp()),
                    history=[{'role': 'system', 'content': system_message}]
                )
            
            # IMPORTANT: Perform agent_id check *inside* the locked section
            if self._storage[thread_id].agent_id != agent_id:
                raise OperationNotAllowedError(f"Thread ID {thread_id} belongs to a different agent: {self._storage[thread_id].agent_id}, the provided agent ID is {agent_id}.")

            self._storage[thread_id].history.extend(content)
            self._storage[thread_id].updated_at = int(datetime.now().timestamp())
            return self._storage[thread_id]

    def clear_context(self, thread_id: str) -> Optional[ContextUnit]:
        thread_lock = self._get_or_create_thread_lock(thread_id)
        with thread_lock: # Acquire lock specific to this thread_id
            if thread_id in self._storage:

                system_message_content = ""
                # Attempt to retrieve existing system message before clearing
                for message in self._storage[thread_id].history:
                    if message['role'] == 'system':
                        system_message_content = message['content']
                        break

                # Preserve agent_id when recreating the ContextUnit
                existing_agent_id = self._storage[thread_id].agent_id

                self._storage[thread_id] = ContextUnit(
                    thread_id=thread_id,
                    agent_id=existing_agent_id, # Crucially preserve the original agent_id
                    history=[{"role": "system", "content": system_message_content}],
                    created_at=int(datetime.now().timestamp()), # Could also preserve original created_at
                    updated_at=int(datetime.now().timestamp())
                )
            else:
                logger.warning(f"Attempted to clear context for non-existent thread ID: {thread_id}")
                return None
            return self._storage[thread_id]
        
    def delete_expired_contexts(self, expiration_time: int) -> int:
        """
        Delete contexts that have not been updated for a specified amount of time.
        
        :param int expiration_time: The time in seconds after which a context is considered expired.
        :return: The number of contexts deleted.
        :rtype: int
        """
        current_time = int(datetime.now().timestamp())
        expired_thread_ids = [
            thread_id for thread_id, context in self._storage.items()
            if current_time - context.updated_at > expiration_time
        ]
        
        with self._main_lock:
            for thread_id in expired_thread_ids:
                if thread_id in self._storage:
                    del self._storage[thread_id]
                    logger.info(f"Deleted expired context for thread ID: {thread_id}")
                else:
                    logger.warning(f"Attempted to delete non-existent context for thread ID: {thread_id}")
                    
        return len(expired_thread_ids)
                    