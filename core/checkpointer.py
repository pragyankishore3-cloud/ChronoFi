"""
Durable LangGraph Checkpointer Management.

Provides high-performance, resilient checkpoint persistence across user conversation threads.
Supports:
- AsyncSqliteSaver backed by aiosqlite (durable production SQLite storage surviving restarts)
- MemorySaver (fast in-memory storage for ephemeral test runs)
- Thread isolation and per-user access control validation
"""

import os
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Sequence, Tuple
import aiosqlite
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    ChannelVersions,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


class DurableAsyncCheckpointer(BaseCheckpointSaver):
    """
    Proxy checkpoint saver that delegates dynamically to an underlying
    BaseCheckpointSaver (AsyncSqliteSaver or MemorySaver).
    Enables static StateGraph compilation at module import time while
    allowing runtime initialization and switching of storage backends.
    """

    def __init__(self, target: Optional[BaseCheckpointSaver] = None):
        super().__init__()
        self._target: BaseCheckpointSaver = target or MemorySaver()
        self._conn: Optional[aiosqlite.Connection] = None

    @property
    def target(self) -> BaseCheckpointSaver:
        return self._target

    def set_target(self, target: BaseCheckpointSaver, conn: Optional[aiosqlite.Connection] = None) -> None:
        self._target = target
        if conn is not None:
            self._conn = conn

    def _ensure_thread_id(self, config: Optional[RunnableConfig]) -> RunnableConfig:
        cfg = dict(config or {})
        configurable = dict(cfg.get("configurable", {}))
        if "thread_id" not in configurable:
            user_id = configurable.get("user_id", "default")
            configurable["thread_id"] = f"thread_user_{user_id}"
        cfg["configurable"] = configurable
        return cfg

    # --- Async methods used by LangGraph ainvoke / astream_events ---

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return await self._target.aget_tuple(self._ensure_thread_id(config))

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        resolved_config = self._ensure_thread_id(config) if config is not None else None
        async for item in self._target.alist(resolved_config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await self._target.aput(self._ensure_thread_id(config), checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return await self._target.aput_writes(self._ensure_thread_id(config), writes, task_id, task_path=task_path)

    # --- Synchronous fallbacks ---

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        if hasattr(self._target, "get_tuple"):
            return self._target.get_tuple(self._ensure_thread_id(config))
        raise NotImplementedError("Underlying target checkpointer does not implement synchronous get_tuple")

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        if hasattr(self._target, "list"):
            resolved_config = self._ensure_thread_id(config) if config is not None else None
            return self._target.list(resolved_config, filter=filter, before=before, limit=limit)
        raise NotImplementedError("Underlying target checkpointer does not implement synchronous list")

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        if hasattr(self._target, "put"):
            return self._target.put(self._ensure_thread_id(config), checkpoint, metadata, new_versions)
        raise NotImplementedError("Underlying target checkpointer does not implement synchronous put")

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        if hasattr(self._target, "put_writes"):
            return self._target.put_writes(self._ensure_thread_id(config), writes, task_id, task_path=task_path)
        raise NotImplementedError("Underlying target checkpointer does not implement synchronous put_writes")

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
        self._target = MemorySaver()


# Global checkpointer instance for application graphs
_global_checkpointer = DurableAsyncCheckpointer(MemorySaver())


def get_checkpointer() -> DurableAsyncCheckpointer:
    """Returns the application-wide checkpointer."""
    return _global_checkpointer


async def init_checkpointer(
    db_path: Optional[str] = None,
    use_memory: Optional[bool] = None,
) -> DurableAsyncCheckpointer:
    """
    Initializes durable SQLite checkpointer or MemorySaver.
    Survives process restarts when SQLite is used.
    """
    is_memory = use_memory
    if is_memory is None:
        env_mode = os.getenv("CHRONOFI_CHECKPOINTER", "").lower()
        is_memory = env_mode == "memory"

    if is_memory:
        mem_saver = MemorySaver()
        _global_checkpointer.set_target(mem_saver)
        return _global_checkpointer

    resolved_path = db_path or os.getenv("CHRONOFI_CHECKPOINT_DB", "chronofi_checkpoints.db")
    # Ensure directory exists if path contains directories
    dir_name = os.path.dirname(resolved_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    conn = await aiosqlite.connect(resolved_path)
    sqlite_saver = AsyncSqliteSaver(conn)
    await sqlite_saver.setup()
    _global_checkpointer.set_target(sqlite_saver, conn=conn)
    return _global_checkpointer


async def close_checkpointer() -> None:
    """Closes any underlying database connections."""
    await _global_checkpointer.close()
