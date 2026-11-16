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

"""Coalescing embedding model: share in-flight work and cache successes.

Concurrent ``embed`` calls for the same text share one underlying provider
request instead of repeating it. Texts that arrive while a batch is still
waiting for its coalescing window join that batch; texts that arrive while
a batch is already executing join the running request. Successful vectors
are kept in a bounded LRU cache for later calls, while failures are never
cached and can be retried. Each call still receives its vectors in its own
original order, with duplicates preserved, and its usage counts each
underlying batch it waited on exactly once.

Cancellation is isolated by construction: a batch runs as an independent
task and waiters only await a shared future, so cancelling one waiter
neither cancels the batch nor disturbs the other waiters.
"""

from __future__ import annotations

import asyncio
import itertools
from collections import OrderedDict
from dataclasses import dataclass

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel


@dataclass(slots=True)
class _PendingText:
    """One distinct text with an outstanding provider batch."""

    text: str
    future: asyncio.Future[EmbeddingVector]
    dispatched: bool = False
    batch_id: int | None = None
    usage: InferenceUsage | None = None


def _sum_optional(values: list[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


class CoalescingEmbeddingModel:
    """Wrap an embedding model with cross-call coalescing and an LRU cache."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int) or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if isinstance(coalesce_ms, bool) or not isinstance(coalesce_ms, (int, float)) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be a non-negative number")  # noqa: TRY003
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")  # noqa: TRY003
        self._base = model
        self._max_batch_size = max_batch_size
        self._coalesce_ms = float(coalesce_ms)
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._inflight: dict[str, _PendingText] = {}
        self._dispatch_handle: asyncio.TimerHandle | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._batch_ids = itertools.count(1)

    @property
    def profile(self):
        """Expose the wrapped model's deployment-fixed profile."""

        return self._base.profile

    async def embed(self, texts: tuple[str, ...]) -> EmbeddingResult:
        """Embed texts, sharing in-flight work and reusing cached vectors."""

        ordered = tuple(texts)
        if not ordered:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))
        vectors, waited = self._register(asyncio.get_running_loop(), ordered)
        if self._coalesce_ms == 0:
            self._flush()
        first_error = await self._collect(waited, vectors)
        if first_error is not None:
            raise first_error
        return EmbeddingResult(
            vectors=tuple(vectors[text] for text in ordered),
            usage=self._aggregate_usage(waited),
        )

    def _register(
        self, loop: asyncio.AbstractEventLoop, ordered: tuple[str, ...]
    ) -> tuple[dict[str, EmbeddingVector], list[_PendingText]]:
        """Synchronously resolve cache hits and join or create pending entries.

        This runs without awaiting so concurrent calls arriving in the same
        event-loop turn share the same pending batches instead of starting
        duplicate work.
        """

        unique = list(dict.fromkeys(ordered))
        vectors: dict[str, EmbeddingVector] = {}
        waited: list[_PendingText] = []
        for text in unique:
            cached = self._cache_get(text)
            if cached is not None:
                vectors[text] = cached
                continue
            entry = self._inflight.get(text)
            if entry is None:
                entry = _PendingText(text=text, future=loop.create_future())
                self._inflight[text] = entry
                if self._coalesce_ms > 0 and self._dispatch_handle is None:
                    self._dispatch_handle = loop.call_later(self._coalesce_ms / 1000, self._flush)
            waited.append(entry)
        return vectors, waited

    async def _collect(self, waited: list[_PendingText], vectors: dict[str, EmbeddingVector]) -> Exception | None:
        """Await each waited entry and return the first batch error, if any."""

        first_error: Exception | None = None
        for entry in waited:
            # Shield the shared future: cancelling this waiter must cancel
            # only its own wait, never the batch other waiters depend on.
            try:
                vectors[entry.text] = await asyncio.shield(entry.future)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if first_error is None:
                    first_error = error
        return first_error

    def _aggregate_usage(self, waited: list[_PendingText]) -> InferenceUsage:
        """Sum each distinct waited batch's usage exactly once."""

        usages: dict[int, InferenceUsage] = {}
        for entry in waited:
            if entry.batch_id is not None and entry.usage is not None:
                usages.setdefault(entry.batch_id, entry.usage)
        batch_usages = list(usages.values())
        return InferenceUsage(
            requests=sum(usage.requests for usage in batch_usages),
            input_tokens=_sum_optional([usage.input_tokens for usage in batch_usages]),
            output_tokens=_sum_optional([usage.output_tokens for usage in batch_usages]),
        )

    def _cache_get(self, text: str) -> EmbeddingVector | None:
        vector = self._cache.get(text)
        if vector is not None:
            self._cache.move_to_end(text)
        return vector

    def _cache_put(self, text: str, vector: EmbeddingVector) -> None:
        if self._cache_size == 0:
            return
        self._cache.pop(text, None)
        self._cache[text] = vector
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _flush(self) -> None:
        """Dispatch every pending text as provider batches of at most the limit."""

        self._dispatch_handle = None
        pending = [entry for entry in self._inflight.values() if not entry.dispatched]
        if not pending:
            return
        for index in range(0, len(pending), self._max_batch_size):
            chunk = pending[index : index + self._max_batch_size]
            for entry in chunk:
                entry.dispatched = True
            task = asyncio.create_task(self._execute_batch(tuple(entry.text for entry in chunk), chunk))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _execute_batch(self, texts: tuple[str, ...], entries: list[_PendingText]) -> None:
        """Run one provider batch and settle its waiters, caching only successes."""

        batch_id = next(self._batch_ids)
        for entry in entries:
            entry.batch_id = batch_id
        try:
            result = await self._base.embed(texts)
        except asyncio.CancelledError as error:
            self._fail_entries(entries, error)
            raise
        except Exception as error:
            self._fail_entries(entries, error)
            return
        if len(result.vectors) != len(texts):
            self._fail_entries(
                entries,
                ValueError(f"embedding model returned {len(result.vectors)} vectors for {len(texts)} texts"),
            )
            return
        for entry, vector in zip(entries, result.vectors, strict=True):
            entry.usage = result.usage
            self._cache_put(entry.text, vector)
        for entry, vector in zip(entries, result.vectors, strict=True):
            self._inflight.pop(entry.text, None)
            if not entry.future.done():
                entry.future.set_result(vector)

    def _fail_entries(self, entries: list[_PendingText], error: BaseException) -> None:
        """Drop failed entries without caching so a later call can retry them."""

        for entry in entries:
            self._inflight.pop(entry.text, None)
            if not entry.future.done():
                entry.future.set_exception(error)


__all__ = ["CoalescingEmbeddingModel"]
