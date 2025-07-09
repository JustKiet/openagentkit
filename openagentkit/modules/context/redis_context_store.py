from openagentkit.core.interfaces.base_context_store import BaseContextStore
from openagentkit.core.models.io.context_unit import ContextUnit # Assume this is a Pydantic model
from openagentkit.core.exceptions import OperationNotAllowedError
from datetime import datetime
from typing import Any, Optional
import logging
from redis import Redis, WatchError

logger = logging.getLogger(__name__)

class RedisContextStore(BaseContextStore):
    _KEY_PREFIX = "context:"

    def __init__(
        self,
        client: Redis,
        ttl: Optional[int] = None
    ) -> None:
        """
        Initialize the RedisContextStore with a Redis client and an optional TTL (time-to-live) for context entries.

        :param redis.Redis client: An instance of Redis client to interact with the Redis database.
        :param Optional[int] ttl: Optional time-to-live in seconds for context entries. If provided, contexts will expire after this duration.
        """
        self._client: Redis = client
        self._ttl: Optional[int] = ttl

    def _get_redis_key(self, thread_id: str) -> str:
        return f"{self._KEY_PREFIX}{thread_id}"

    def _save_context_to_redis(self, context: ContextUnit) -> None:
        """Serializes and saves ContextUnit to Redis, applying TTL if configured."""
        key = self._get_redis_key(context.thread_id)
        self._client.set(key, context.model_dump_json())
        if self._ttl is not None:
            self._client.expire(key, self._ttl)

    def _get_context_from_redis(self, thread_id: str) -> Optional[ContextUnit]:
        """Retrieves and deserializes ContextUnit from Redis."""
        key = self._get_redis_key(thread_id)
        serialized_data = self._client.get(key)
        if serialized_data:
            if isinstance(serialized_data, bytes):
                    serialized_data = serialized_data.decode('utf-8')
            elif not isinstance(serialized_data, str):
                raise ValueError(f"Unexpected type for serialized data: {type(serialized_data)}. Expected bytes or str.")
            return ContextUnit.model_validate_json(serialized_data)
        return None

    def get_system_message(self, thread_id: str) -> str:
        context = self._get_context_from_redis(thread_id)
        if context:
            for message in context.history:
                if message['role'] == 'system':
                    return message['content']
        return ""

    def update_system_message(self, thread_id: str, agent_id: str, system_message: str) -> None:
        key = self._get_redis_key(thread_id)

        with self._client.pipeline() as pipe: # type: ignore[no-redef]
            while True:
                try:
                    pipe.watch(key)

                    context = None
                    serialized_data = pipe.get(key)
                    if serialized_data:
                        if isinstance(serialized_data, bytes):
                            serialized_data = serialized_data.decode('utf-8')
                        elif not isinstance(serialized_data, str):
                            raise ValueError(f"Unexpected type for serialized data: {type(serialized_data)}. Expected bytes or str.")

                        context = ContextUnit.model_validate_json(serialized_data)

                    if context is None:
                        pipe.unwatch()
                        self.init_context(thread_id, agent_id, system_message)
                        return

                    if context.agent_id != agent_id:
                        pipe.unwatch()
                        raise OperationNotAllowedError(
                            f"Thread ID {thread_id} belongs to a different agent: {context.agent_id}, provided ID: {agent_id}."
                        )

                    for message in context.history:
                        if message['role'] == 'system':
                            message['content'] = system_message
                            break
                    else:
                        context.history.insert(0, {"role": "system", "content": system_message})

                    context.updated_at = int(datetime.now().timestamp())

                    pipe.multi()
                    pipe.set(key, context.model_dump_json())
                    if self._ttl is not None:
                        pipe.expire(key, self._ttl) # Reapply TTL
                    pipe.execute()
                    break
                except WatchError:
                    logger.warning(f"WatchError on update_system_message for {thread_id}. Retrying.")

    def init_context(
        self,
        thread_id: str,
        agent_id: str,
        system_message: str
    ) -> ContextUnit:
        key = self._get_redis_key(thread_id)

        new_context = ContextUnit(
            thread_id=thread_id,
            agent_id=agent_id,
            history=[{"role": "system", "content": system_message}],
            created_at=int(datetime.now().timestamp()),
            updated_at=int(datetime.now().timestamp())
        )

        # Using set with nx=True and ex for atomicity
        set_result = self._client.set(
            key,
            new_context.model_dump_json(),
            nx=True,
            ex=self._ttl if self._ttl is not None else None # Apply TTL here
        )

        if set_result is None: # set with nx=True returns None if key already exists
            # Check if agent_id does not match an existing context
            existing_context = self._get_context_from_redis(thread_id)
            if existing_context:
                if existing_context.agent_id != agent_id:
                    logger.error(f"Context with thread ID {thread_id} already exists for agent {existing_context.agent_id}.")
                    raise OperationNotAllowedError(f"Context with thread ID {thread_id} already exists.")
            
                if system_message != existing_context.system_message:
                    self.update_system_message(thread_id, agent_id, system_message) # Update system message if it differs
            
                return existing_context
        
        return new_context

    def get_context(self, thread_id: str) -> Optional[ContextUnit]:
        return self._get_context_from_redis(thread_id)

    def get_agent_context(self, agent_id: str) -> dict[str, ContextUnit]:
        """Retrieves all contexts associated with a specific agent ID."""
        contexts: dict[str, ContextUnit] = {}
        keys: list[str | bytes] = self._client.keys(f"{self._KEY_PREFIX}*") # type: ignore[no-redef]
        # Ensure keys are str for further processing
        str_keys = [key.decode('utf-8') if isinstance(key, bytes) else key for key in keys]

        for key in str_keys:
            serialized_data = self._client.get(key)
            if serialized_data:
                if isinstance(serialized_data, bytes):
                    serialized_data = serialized_data.decode('utf-8')
                elif not isinstance(serialized_data, str):
                    raise ValueError(f"Unexpected type for serialized data: {type(serialized_data)}. Expected bytes or str.")

                context = ContextUnit.model_validate_json(serialized_data)
                if context.agent_id == agent_id:
                    contexts[context.thread_id] = context

        return contexts

    def add_context(
        self, 
        thread_id: str, 
        agent_id: str, 
        content: dict[str, Any],
        system_message: Optional[str] = None    
    ) -> ContextUnit:
        key = self._get_redis_key(thread_id)

        with self._client.pipeline() as pipe: # type: ignore[no-redef]
            while True:
                try:
                    pipe.watch(key)

                    context = None
                    serialized_data = pipe.get(key)
                    if serialized_data:
                        if isinstance(serialized_data, bytes):
                            serialized_data = serialized_data.decode('utf-8')
                        elif not isinstance(serialized_data, str):
                            raise ValueError(f"Unexpected type for serialized data: {type(serialized_data)}. Expected bytes or str.")
                        context = ContextUnit.model_validate_json(serialized_data)

                    if context is None:
                        pipe.unwatch()
                        # If context doesn't exist, create it with initial history and TTL
                        if not system_message:
                            raise ValueError("System message must be provided when initializing a new context.")
                        
                        new_history = [{"role": "system", "content": system_message}]
                        new_history.append(content) if content else None

                        new_context = ContextUnit(
                            thread_id=thread_id,
                            agent_id=agent_id,
                            created_at=int(datetime.now().timestamp()),
                            updated_at=int(datetime.now().timestamp()),
                            history=new_history # Add the initial content here
                        )
                        # Use set with nx=True and ex
                        set_result = self._client.set(
                            key,
                            new_context.model_dump_json(),
                            nx=True,
                            ex=self._ttl if self._ttl is not None else None
                        )
                        if set_result is None: # Another process might have created it
                            continue # Retry the watch
                        context = new_context
                        # Since we just created it with the content, we can break
                        break

                    if context.agent_id != agent_id:
                        pipe.unwatch()
                        raise OperationNotAllowedError(
                            f"Thread ID {thread_id} belongs to a different agent: {context.agent_id}, provided ID: {agent_id}."
                        )

                    context.history.append(content)
                    context.updated_at = int(datetime.now().timestamp())

                    pipe.multi()
                    pipe.set(key, context.model_dump_json())
                    if self._ttl is not None:
                        pipe.expire(key, self._ttl) # Reapply TTL
                    pipe.execute()
                    break
                except WatchError:
                    logger.warning(f"WatchError on add_context for {thread_id}. Retrying.")

        return context

    def extend_context(
        self, 
        thread_id: str, 
        agent_id: str, 
        content: list[dict[str, Any]],
        system_message: Optional[str] = None
    ) -> ContextUnit:
        key = self._get_redis_key(thread_id)

        with self._client.pipeline() as pipe: # type: ignore[no-redef]
            while True:
                try:
                    pipe.watch(key)

                    context = None
                    serialized_data = pipe.get(key)
                    if serialized_data:
                        if isinstance(serialized_data, bytes):
                            serialized_data = serialized_data.decode('utf-8')
                        elif not isinstance(serialized_data, str):
                            raise ValueError(f"Unexpected type for serialized data: {type(serialized_data)}. Expected bytes or str.")

                        context = ContextUnit.model_validate_json(serialized_data)

                    if context is None:
                        pipe.unwatch()
                        # If context doesn't exist, create it with initial history and TTL
                        if not system_message:
                            raise ValueError("System message must be provided when initializing a new context.")
                        
                        new_history = [{"role": "system", "content": system_message}]
                        new_history.extend(content) if content else None

                        new_context = ContextUnit(
                            thread_id=thread_id,
                            agent_id=agent_id,
                            created_at=int(datetime.now().timestamp()),
                            updated_at=int(datetime.now().timestamp()),
                            history=new_history # Add the initial content here
                        )
                        # Use set with nx=True and ex
                        set_result = self._client.set(
                            key,
                            new_context.model_dump_json(),
                            nx=True,
                            ex=self._ttl if self._ttl is not None else None
                        )
                        if set_result is None: # Another process might have created it
                            continue # Retry the watch
                        context = new_context
                        # Since we just created it with the content, we can break
                        break

                    if context.agent_id != agent_id:
                        pipe.unwatch()
                        raise OperationNotAllowedError(
                            f"Thread ID {thread_id} belongs to a different agent: {context.agent_id}, provided ID: {agent_id}."
                        )

                    context.history.extend(content)
                    context.updated_at = int(datetime.now().timestamp())

                    pipe.multi()
                    pipe.set(key, context.model_dump_json())
                    if self._ttl is not None:
                        pipe.expire(key, self._ttl) # Reapply TTL
                    pipe.execute()
                    break
                except WatchError:
                    logger.warning(f"WatchError on extend_context for {thread_id}. Retrying.")

        return context

    def clear_context(self, thread_id: str) -> Optional[ContextUnit]:
        key = self._get_redis_key(thread_id)

        with self._client.pipeline() as pipe: # type: ignore[no-redef]
            while True:
                try:
                    pipe.watch(key)

                    context = self._get_context_from_redis(thread_id)
                    if context is None:
                        pipe.unwatch()
                        logger.warning(f"Attempted to clear context for non-existent ID: {thread_id}")
                        return None

                    system_message_content = ""
                    for message in context.history:
                        if message['role'] == 'system':
                            system_message_content = message['content']
                            break

                    cleared_context = ContextUnit(
                        thread_id=thread_id,
                        agent_id=context.agent_id,
                        history=[{"role": "system", "content": system_message_content}],
                        created_at=context.created_at,
                        updated_at=int(datetime.now().timestamp())
                    )

                    pipe.multi()
                    pipe.set(key, cleared_context.model_dump_json())
                    if self._ttl is not None:
                        pipe.expire(key, self._ttl) # Reapply TTL
                    pipe.execute()
                    break
                except WatchError:
                    logger.warning(f"WatchError on clear_context for {thread_id}. Retrying.")

        return cleared_context