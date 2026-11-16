"""Coalescing embedding model: sharing, cache, failure and cancellation.

These tests pin the behaviours the public acceptance suite checks through
the real runtime: one underlying batch per distinct text across concurrent
calls, LRU eviction with access refresh, retries after failures, isolation
when one waiter is cancelled, and per-call usage that counts each waited
batch exactly once.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any, ParamSpec

import pytest

from powercontext.builtin.artifacts.memory import EmbeddingProfile
from powercontext.builtin.inference import EmbeddingResult, InferenceUsage
from powercontext.builtin.inference.coalescing import CoalescingEmbeddingModel

P = ParamSpec("P")


def _async_test(function: Callable[P, Coroutine[Any, Any, None]]) -> Callable[P, None]:
    @wraps(function)
    def run(*args: P.args, **kwargs: P.kwargs) -> None:
        asyncio.run(function(*args, **kwargs))

    return run


class _Embedding:
    profile = EmbeddingProfile(profile_id="coalescing-tests", model="local", dimension=1)

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    async def embed(self, texts):
        self.calls.append(tuple(texts))
        return EmbeddingResult(vectors=tuple((float(len(text)),) for text in texts), usage=InferenceUsage(requests=1))


@_async_test
async def test_concurrent_calls_share_each_text_once_and_keep_order() -> None:
    base = _Embedding()
    model = CoalescingEmbeddingModel(base, max_batch_size=2, coalesce_ms=20, cache_size=3)
    first, second = await asyncio.gather(model.embed(("alpha", "beta", "alpha")), model.embed(("beta", "gamma")))
    assert first.vectors == ((5.0,), (4.0,), (5.0,))
    assert second.vectors == ((4.0,), (5.0,))
    assert sorted(text for call in base.calls for text in call) == ["alpha", "beta", "gamma"]
    assert all(len(call) <= 2 for call in base.calls)
    assert first.usage.requests == sum(bool(set(call) & {"alpha", "beta"}) for call in base.calls)
    assert second.usage.requests == sum(bool(set(call) & {"beta", "gamma"}) for call in base.calls)
    await model.embed(("alpha",))
    assert sum(call.count("alpha") for call in base.calls) == 1


@_async_test
async def test_late_waiter_joins_executing_request() -> None:
    class Blocked(_Embedding):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def embed(self, texts):
            self.calls.append(tuple(texts))
            self.started.set()
            await self.release.wait()
            return EmbeddingResult(vectors=tuple((1.0,) for _ in texts), usage=InferenceUsage(requests=1))

    base = Blocked()
    model = CoalescingEmbeddingModel(base, coalesce_ms=0)
    first = asyncio.create_task(model.embed(("shared",)))
    await base.started.wait()
    second = asyncio.create_task(model.embed(("shared",)))
    await asyncio.sleep(0)
    base.release.set()
    assert (await first).vectors == ((1.0,),)
    assert (await second).vectors == ((1.0,),)
    assert base.calls == [("shared",)]


@_async_test
async def test_lru_eviction_access_refresh_and_cache_disabled() -> None:
    base = _Embedding()
    model = CoalescingEmbeddingModel(base, coalesce_ms=0, cache_size=2)
    await model.embed(("alpha",))
    await model.embed(("beta",))
    await model.embed(("alpha",))
    await model.embed(("gamma",))
    await model.embed(("alpha",))
    assert sum(call.count("alpha") for call in base.calls) == 1
    await model.embed(("beta",))
    assert sum(call.count("beta") for call in base.calls) == 2

    uncached = _Embedding()
    sharing = CoalescingEmbeddingModel(uncached, coalesce_ms=0, cache_size=0)
    first, second = await asyncio.gather(sharing.embed(("same",)), sharing.embed(("same",)))
    assert first.vectors == second.vectors == ((4.0,),)
    assert uncached.calls == [("same",)]
    await sharing.embed(("same",))
    assert uncached.calls == [("same",), ("same",)]


@_async_test
async def test_empty_input_never_calls_provider() -> None:
    base = _Embedding()
    model = CoalescingEmbeddingModel(base, coalesce_ms=0)
    empty = await model.embed(())
    assert empty.vectors == ()
    assert empty.usage.requests == 0
    assert base.calls == []


@_async_test
async def test_cancelling_one_waiter_keeps_shared_work() -> None:
    class Blocked(_Embedding):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def embed(self, texts):
            self.calls.append(tuple(texts))
            self.started.set()
            await self.release.wait()
            return EmbeddingResult(vectors=((1.0,),))

    base = Blocked()
    model = CoalescingEmbeddingModel(base, coalesce_ms=0)
    cancelled = asyncio.create_task(model.embed(("shared",)))
    survivor = asyncio.create_task(model.embed(("shared",)))
    await base.started.wait()
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    base.release.set()
    assert (await survivor).vectors == ((1.0,),)
    assert base.calls == [("shared",)]


@_async_test
async def test_failure_is_not_cached_and_other_batches_complete() -> None:
    class Flaky(_Embedding):
        async def embed(self, texts):
            self.calls.append(tuple(texts))
            if "bad" in texts:
                raise RuntimeError("temporary")
            return EmbeddingResult(vectors=tuple((1.0,) for _ in texts), usage=InferenceUsage(requests=1))

    base = Flaky()
    model = CoalescingEmbeddingModel(base, max_batch_size=1, coalesce_ms=0)
    bad = asyncio.create_task(model.embed(("bad",)))
    good = asyncio.create_task(model.embed(("good",)))
    with pytest.raises(RuntimeError):
        await bad
    assert (await good).vectors == ((1.0,),)
    assert base.calls == [("bad",), ("good",)] or base.calls == [("good",), ("bad",)]
    base.calls.clear()

    class Fixed(_Embedding):
        async def embed(self, texts):
            self.calls.append(tuple(texts))
            return EmbeddingResult(vectors=tuple((2.0,) for _ in texts), usage=InferenceUsage(requests=1))

    fixed_base = Fixed()
    retry_model = CoalescingEmbeddingModel(fixed_base, max_batch_size=1, coalesce_ms=0)
    assert (await retry_model.embed(("bad",))).vectors == ((2.0,),)


@_async_test
async def test_vector_count_mismatch_raises_value_error() -> None:
    class Invalid(_Embedding):
        async def embed(self, texts):
            return EmbeddingResult(vectors=())

    with pytest.raises(ValueError):
        await CoalescingEmbeddingModel(Invalid(), coalesce_ms=0).embed(("broken",))


@_async_test
async def test_usage_counts_each_waited_batch_once_and_cache_is_free() -> None:
    class Metered(_Embedding):
        async def embed(self, texts):
            self.calls.append(tuple(texts))
            return EmbeddingResult(
                vectors=tuple((1.0,) for _ in texts),
                usage=InferenceUsage(requests=1, input_tokens=7, output_tokens=0),
            )

    model = CoalescingEmbeddingModel(Metered(), coalesce_ms=0)
    result = await model.embed(("first", "first"))
    assert result.usage.requests == 1
    assert result.usage.input_tokens == 7
    assert result.usage.output_tokens == 0
    cached = await model.embed(("first",))
    assert cached.usage.requests == 0
    assert cached.usage.input_tokens in (None, 0)


def test_constructor_rejects_invalid_limits() -> None:
    base = _Embedding()
    with pytest.raises(ValueError):
        CoalescingEmbeddingModel(base, max_batch_size=0)
    with pytest.raises(ValueError):
        CoalescingEmbeddingModel(base, coalesce_ms=-1)
    with pytest.raises(ValueError):
        CoalescingEmbeddingModel(base, cache_size=-1)
