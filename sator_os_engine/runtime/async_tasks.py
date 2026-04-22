from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any, TypeVar

log = logging.getLogger("sator")

T = TypeVar("T")


def spawn_task_logged(coro: Coroutine[Any, Any, T], *, label: str) -> asyncio.Task[T]:
    """Schedule a coroutine and log any unhandled exception from its completion (fire-and-forget)."""

    task: asyncio.Task[T] = asyncio.create_task(coro)

    def _done(t: asyncio.Task) -> None:
        if t.cancelled():
            return
        try:
            exc = t.exception()
        except (asyncio.CancelledError, asyncio.InvalidStateError):
            return
        if exc is not None:
            log.error("async_background_task_failed label=%s", label, exc_info=exc)

    task.add_done_callback(_done)
    return task
