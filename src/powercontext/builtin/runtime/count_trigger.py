# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Count-based admission for Memory and Topic Memory processing.

A scope's pending count is the number of journal entries after the Family's
last successful cursor. When that count reaches the configured threshold, the
Runtime automatically runs one existing processing call: one Memory window,
or one accepted Topic Memory wave. The count condition is an OR with the
existing time condition, and explicit flush bypasses the threshold entirely.

The driver only decides *when* to run work. It never changes how a window is
consumed: Memory re-checks the remaining count after each successful window
and stops below the threshold, while an accepted Topic Memory wave completes
its own windows without re-applying the threshold. A failed call never
advances the success cursor, so the pending count survives a restart and the
startup scan re-admits it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from powercontext.builtin.artifacts.topic_memory import TOPIC_MEMORY_SOURCE_WINDOW_BINDING
from powercontext.builtin.triggers import SOURCE_WINDOW_TRIGGER_NAME

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import BuiltinRuntime

logger = logging.getLogger(__name__)


class SourceCountTrigger:
    """Drive count-threshold admission for one built-in runtime."""

    def __init__(self, runtime: BuiltinRuntime, /) -> None:
        self._runtime = runtime
        self._tasks: set[asyncio.Task[None]] = set()
        self._locks: dict[str, asyncio.Lock] = {}
        self._scan_started = False

    @property
    def enabled(self) -> bool:
        """Return whether any Family has a count threshold configured."""

        return (
            self._runtime.memory_source_count_threshold is not None
            or self._runtime.topic_memory_source_count_threshold is not None
        )

    def notify(self, scope_id: str, /) -> None:
        """Schedule a count re-check for one scope after a journal write.

        Fire-and-forget: capture and artifact writes hold the scope lock, and
        the re-check itself takes the same lock through flush, so the work
        must run after the caller releases it. A check that finds nothing to
        do returns immediately; concurrent checks for one scope serialize on
        a per-scope lock and re-read the count, so no notification is lost.
        """

        if not self.enabled or self._runtime._closing or self._runtime._closed:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._run_scope(scope_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def start_scan(self) -> None:
        """Re-check every journaled scope once, for work left by a restart."""

        if not self.enabled or self._scan_started or self._runtime._closing or self._runtime._closed:
            return
        self._scan_started = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._scan_all())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def shutdown(self) -> None:
        """Cancel checks that never started; in-flight work already finished."""

        pending = [task for task in self._tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _scan_all(self) -> None:
        try:
            async with self._runtime._operation():
                scope_ids = await self._runtime._provider.scope_ids()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._log(error, "count trigger startup scan")
            return
        for scope_id in scope_ids:
            if self._runtime._closing or self._runtime._closed:
                return
            self.notify(scope_id)

    async def _run_scope(self, scope_id: str, /) -> None:
        if self._runtime._closing or self._runtime._closed:
            return
        try:
            async with self._runtime._operation():
                lock = self._locks.setdefault(scope_id, asyncio.Lock())
                async with lock:
                    await self._check_scope(scope_id)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._log(error, f"count trigger check for scope {scope_id}")

    async def _check_scope(self, scope_id: str, /) -> None:
        threshold = self._runtime.memory_source_count_threshold
        if threshold is not None:
            await self._drive_memory(scope_id, threshold)
        threshold = self._runtime.topic_memory_source_count_threshold
        if threshold is not None:
            await self._drive_topic_memory(scope_id, threshold)

    async def _drive_memory(self, scope_id: str, threshold: int, /) -> None:
        """Run Memory windows while the remaining count still meets the threshold."""

        while True:
            try:
                pending = await self._runtime._provider.pending_source_count(scope_id, SOURCE_WINDOW_TRIGGER_NAME)
            except Exception as error:
                self._log(error, f"count trigger pending read for scope {scope_id}")
                return
            if pending < threshold:
                return
            try:
                result = await self._runtime.memory.for_scope(scope_id).flush()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # A failed call must not consume the success cursor, and the
                # pending count stays at or above the threshold, so the next
                # capture, time tick, or restart re-admits the same work.
                self._log(error, f"count-triggered Memory window for scope {scope_id}")
                return
            if not result.processed:
                return

    async def _drive_topic_memory(self, scope_id: str, threshold: int, /) -> None:
        """Accept one Topic Memory wave when the count meets the threshold."""

        if not self._runtime._topic_memory_processing_available or self._runtime._topic_memory_flush is None:
            return
        try:
            pending = await self._runtime._provider.pending_source_count(scope_id, TOPIC_MEMORY_SOURCE_WINDOW_BINDING)
        except Exception as error:
            self._log(error, f"count trigger pending read for scope {scope_id}")
            return
        if pending < threshold:
            return
        try:
            accepted = await self._runtime._topic_memory_flush(scope_id)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._log(error, f"count-triggered Topic Memory wave for scope {scope_id}")
            return
        if accepted and self._runtime.artifact_processing_supervisor is not None:
            try:
                self._runtime.artifact_processing_supervisor.wake(TOPIC_MEMORY_SOURCE_WINDOW_BINDING)
            except Exception as error:
                logger.warning(
                    "Topic Memory supervisor wake failed after count trigger acceptance",
                    exc_info=error,
                    extra={"event": "topic_memory.count_trigger.wake_failed", "outcome": "accepted"},
                )

    def _log(self, error: Exception, what: str, /) -> None:
        if self._runtime._closing or self._runtime._closed:
            logger.debug("Skipped %s while the runtime is closing: %s", what, error)
            return
        logger.warning("Count trigger failed during %s", what, exc_info=error)
