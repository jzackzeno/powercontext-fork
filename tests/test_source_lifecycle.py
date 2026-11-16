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

"""Focused behavior tests for Source lifecycle propagation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any, ParamSpec
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import MemoryEntryInput
from powercontext.builtin.artifacts.memory.service import MemoryService
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.records import ArtifactWrite
from powercontext.builtin.runtime import (
    BuiltinConfig,
    CaptureSource,
    GetMemoryEntryRequest,
    RememberMemoryRequest,
    SearchMemoryRequest,
    SearchTopicMemoryRequest,
    open_builtin_runtime,
)
from powercontext.builtin.runtime.source_lifecycle import SourceLifecycleError, SourceLifecycleEvent
from powercontext.builtin.scope import ScopeDraft

P = ParamSpec("P")


def _async_test(function: Callable[P, Coroutine[Any, Any, None]]) -> Callable[P, None]:
    @wraps(function)
    def run(*args: P.args, **kwargs: P.kwargs) -> None:
        asyncio.run(function(*args, **kwargs))

    return run


def _event(node_id: str, sequence: int, available: bool) -> SourceLifecycleEvent:
    return SourceLifecycleEvent(
        node_id=node_id, revision=3, sequence=sequence, available=available, reason="test lifecycle"
    )


async def _build(runtime, *, suffix: str = "own"):
    scope = await runtime.scopes.create(
        ScopeDraft(title="Lifecycle", summary="Dependency evidence", idempotency_key=suffix)
    )
    scope_id = scope.scope_id
    lifecycle = runtime.lifecycle.for_scope(scope_id)
    roots = {}
    for number in range(2):
        node = f"root{number}"
        receipt = await runtime.sources.for_scope(scope_id).capture(
            CaptureSource(source_id=node, content=f"Origin {node}", metadata={})
        )
        roots[node] = receipt.source_ref
        await lifecycle.register_source(node, receipt.source_ref, revision=3)
    citations = dict(roots)
    await runtime.memory.for_scope(scope_id).remember(
        RememberMemoryRequest(
            entries=(
                MemoryEntryInput(kind="fact", text="both fixture evidence"),
                MemoryEntryInput(kind="fact", text="first fixture evidence"),
            )
        )
    )
    # remember() returns only the last changed entry; read both via list().
    entries = await runtime.memory.for_scope(scope_id).list(include_inactive=True)
    by_text = {record.entry.text: record.citation for record in entries.entries}
    await lifecycle.register_memory("both", by_text["both fixture evidence"], parents=("root0", "root1"), policy="all")
    await lifecycle.register_memory("first", by_text["first fixture evidence"], parents=("root0",), policy="any")
    citations["both"] = by_text["both fixture evidence"]
    citations["first"] = by_text["first fixture evidence"]
    artifact = await runtime.records.for_scope(scope_id).create_artifact(
        "topic-memory",
        ArtifactWrite(
            content={"title": "topic", "summary": "topic supported summary", "detail": "topic detailed evidence"}
        ),
    )
    reference = ArtifactRef(family="topic-memory", artifact_id=artifact.artifact_id, revision=artifact.revision)
    await lifecycle.register_topic("topic", reference, parents=("both",), policy="all")
    citations["topic"] = reference
    return scope_id, lifecycle, citations


@_async_test
async def test_event_field_validation() -> None:
    with pytest.raises(ValidationError):
        SourceLifecycleEvent(node_id="x", revision=0, sequence=1, available=False, reason="r")
    with pytest.raises(ValidationError):
        SourceLifecycleEvent(node_id="x", revision=1, sequence=0, available=False, reason="r")
    with pytest.raises(ValidationError):
        SourceLifecycleEvent(node_id=" ", revision=1, sequence=1, available=False, reason="r")
    with pytest.raises(ValidationError):
        SourceLifecycleEvent(node_id="x", revision=1, sequence=1, available=False, reason="  ")
    with pytest.raises(ValidationError):
        SourceLifecycleEvent(node_id="x", revision=1, sequence=1, available=False, reason="r", extra=True)
    with pytest.raises(ValidationError):
        SourceLifecycleEvent(node_id="x", revision=True, sequence=1, available=False, reason="r")
    event = SourceLifecycleEvent(node_id="x", revision=1, sequence=1, available=False, reason="r")
    assert event.node_id == "x"


@_async_test
async def test_registration_validation() -> None:
    config = BuiltinConfig(database=SQLiteConfig())
    async with open_builtin_runtime(config) as runtime:
        _scope_id, lifecycle, citations = await _build(runtime)
        with pytest.raises(SourceLifecycleError):
            await lifecycle.register_memory("dup", citations["both"], parents=("root0", "root0"), policy="all")
        with pytest.raises(SourceLifecycleError):
            await lifecycle.register_memory("empty", citations["both"], parents=(), policy="all")
        with pytest.raises(SourceLifecycleError):
            await lifecycle.register_memory("missing", citations["both"], parents=("nope",), policy="all")
        with pytest.raises(SourceLifecycleError):
            await lifecycle.register_memory("self", citations["both"], parents=("self",), policy="all")
        with pytest.raises(SourceLifecycleError):
            await lifecycle.register_memory("bad-policy", citations["both"], parents=("root0",), policy="most")
        with pytest.raises(SourceLifecycleError):
            await lifecycle.register_memory("both", citations["first"], parents=("root0",), policy="all")
        # Identical re-registration is a no-op.
        await lifecycle.register_memory("both", citations["both"], parents=("root0", "root1"), policy="all")
        with pytest.raises(SourceLifecycleError):
            await lifecycle.apply(_event("missing", 1, False))
        with pytest.raises(SourceLifecycleError):
            await lifecycle.apply(
                SourceLifecycleEvent(node_id="root0", revision=2, sequence=1, available=False, reason="r")
            )
        with pytest.raises(SourceLifecycleError):
            await lifecycle.resolve("missing", purpose="generation")
        with pytest.raises(SourceLifecycleError):
            await lifecycle.resolve("root0", purpose="unknown")
        assert await lifecycle.resolve("root0", purpose="audit") == citations["root0"]


@_async_test
async def test_withdrawal_and_restore_propagate_to_search() -> None:
    config = BuiltinConfig(database=SQLiteConfig())
    async with open_builtin_runtime(config) as runtime:
        scope_id, lifecycle, citations = await _build(runtime)
        memory = runtime.memory.for_scope(scope_id)
        # Withdrawing root0 takes out "both" (all of root0, root1), "first"
        # (any of root0), and "topic" (all of "both"); only root1 survives.
        assert await lifecycle.apply(_event("root0", 1, False)) == ["root1"]
        states = {
            record.entry.text.split()[0]: record.state for record in (await memory.list(include_inactive=True)).entries
        }
        assert states == {"both": "inactive", "first": "inactive"}
        page = await memory.search(SearchMemoryRequest(query="both", mode="fts", limit=10))
        assert not any(hit.text == "both fixture evidence" for hit in page.hits)
        topics = await runtime.topic_memory.for_scope(scope_id).search(
            SearchTopicMemoryRequest(query="topic", limit=20)
        )
        assert not topics.hits
        original = await memory.get(GetMemoryEntryRequest(citation=citations["both"]))
        assert original.entry.text == "both fixture evidence"
        with pytest.raises(SourceLifecycleError):
            await lifecycle.resolve("both", purpose="generation")
        assert await lifecycle.resolve("both", purpose="audit") == citations["both"]
        assert await lifecycle.apply(_event("root0", 2, True)) == ["both", "first", "root0", "root1", "topic"]
        page = await memory.search(SearchMemoryRequest(query="both", mode="fts", limit=10))
        assert any(hit.text == "both fixture evidence" for hit in page.hits)
        topics = await runtime.topic_memory.for_scope(scope_id).search(
            SearchTopicMemoryRequest(query="topic", limit=20)
        )
        assert bool(topics.hits)


@_async_test
async def test_failed_propagation_retries_to_completion() -> None:
    config = BuiltinConfig(database=SQLiteConfig())
    async with open_builtin_runtime(config) as runtime:
        scope_id, lifecycle, _citations = await _build(runtime)
        original_forget = MemoryService.forget
        calls = {"count": 0}

        async def flaky_forget(self, memory, *, entries, reason=None):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError
            return await original_forget(self, memory, entries=entries, reason=reason)

        with patch.object(MemoryService, "forget", flaky_forget), pytest.raises(RuntimeError):
            await lifecycle.apply(_event("root0", 1, False))
        # The failed event was not recorded, so the same event retries cleanly.
        assert await lifecycle.apply(_event("root0", 1, False)) == ["root1"]
        states = {
            record.entry.text.split()[0]: record.state
            for record in (await runtime.memory.for_scope(scope_id).list(include_inactive=True)).entries
        }
        assert states == {"both": "inactive", "first": "inactive"}
        # A successful replay adds no new Memory revision.
        before = (await runtime.memory.for_scope(scope_id).list()).memory_ref
        assert await lifecycle.apply(_event("root0", 1, False)) == ["root1"]
        assert (await runtime.memory.for_scope(scope_id).list()).memory_ref == before


@_async_test
async def test_restart_keeps_state_and_replay_is_a_noop(tmp_path) -> None:
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'lifecycle.db'}"))
    async with open_builtin_runtime(config) as runtime:
        scope_id, lifecycle, _ = await _build(runtime)
        await lifecycle.apply(_event("root0", 1, False))
    # Reopening while topics are still withdrawn must not trip the Topic
    # Memory storage invariant: a lifecycle withdrawal is a deliberate
    # inactive head, not corruption.
    async with open_builtin_runtime(config) as runtime:
        lifecycle = runtime.lifecycle.for_scope(scope_id)
        memory = runtime.memory.for_scope(scope_id)
        states = {
            record.entry.text.split()[0]: record.state
            for record in (await memory.list(include_inactive=True)).entries
        }
        assert states == {"both": "inactive", "first": "inactive"}
        page = await memory.search(SearchMemoryRequest(query="both", mode="fts", limit=10))
        assert not any(hit.text == "both fixture evidence" for hit in page.hits)
        before = (await memory.list()).memory_ref
        assert await lifecycle.apply(_event("root0", 1, False)) == ["root1"]
        assert (await memory.list()).memory_ref == before
        with pytest.raises(SourceLifecycleError):
            await lifecycle.apply(_event("root0", 1, True))
        assert await lifecycle.apply(_event("root0", 2, True)) == ["both", "first", "root0", "root1", "topic"]
        page = await memory.search(SearchMemoryRequest(query="both", mode="fts", limit=10))
        assert any(hit.text == "both fixture evidence" for hit in page.hits)


@_async_test
async def test_scopes_are_isolated() -> None:
    config = BuiltinConfig(database=SQLiteConfig())
    async with open_builtin_runtime(config) as runtime:
        first_id, first, _ = await _build(runtime, suffix="first")
        second_id, second, _ = await _build(runtime, suffix="second")
        assert first_id != second_id
        await first.apply(_event("root0", 1, False))
        assert await second.apply(_event("root0", 1, False)) == ["root1"]
        assert await first.apply(_event("root0", 1, False)) == ["root1"]
