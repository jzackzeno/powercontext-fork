"""Pi v3 session history: branch parsing, Source identity and incremental import.

These tests pin the behaviours the public acceptance suite checks through
the real runtime: the text each visible event produces, the stable identity
derived only from session and event ids, and the connector's checkpoint
rules (shared ancestors are not resubmitted, a rewritten imported event
fails the run, a failed run keeps the previous checkpoint).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, cast

import pytest
from pydantic import JsonValue

from powercontext.sources import (
    SourceDefinitionRegistry,
    project_source_for_transport,
)
from powercontext.sources.connectors import ConnectorBinding, ConnectorLifecycle
from powercontext.sources.models import SourceRef
from powercontext.sources.pi_session import (
    PI_EVENT_SOURCE_DEFINITION,
    PiEventInput,
    PiSessionConnector,
    PiSessionInput,
    parse_pi_session,
    pi_event_source_id,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY

P = ParamSpec("P")


def _async_test(function: Callable[P, Coroutine[Any, Any, None]]) -> Callable[P, None]:
    @wraps(function)
    def run(*args: P.args, **kwargs: P.kwargs) -> None:
        asyncio.run(function(*args, **kwargs))

    return run


def _session(records, leaf=None, session_id="session-1"):
    header = {"type": "session", "version": 3, "id": session_id}
    raw = "\n".join(json.dumps(record) for record in [header, *records]).encode()
    return parse_pi_session(PiSessionInput(content=raw, leaf_id=leaf))


def _branch_records():
    return [
        {"type": "message", "id": "a", "parentId": None, "message": {"role": "user", "content": "alpha"}},
        {"type": "message", "id": "b", "parentId": "a", "message": {"role": "assistant", "content": "other"}},
        {"type": "compaction", "id": "c", "parentId": "a", "summary": "retained summary"},
        {"type": "branch_summary", "id": "d", "parentId": "c", "summary": "branch summary"},
        {
            "type": "message",
            "id": "e",
            "parentId": "d",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "private reasoning"},
                    {"type": "toolCall", "name": "read", "id": "call-1", "arguments": {"path": "a.txt"}},
                ],
            },
        },
        {
            "type": "message",
            "id": "f",
            "parentId": "e",
            "message": {
                "role": "toolResult",
                "toolName": "read",
                "toolCallId": "call-1",
                "content": [{"type": "text", "text": "file contents"}],
            },
        },
    ]


def test_branch_selection_keeps_tool_and_summary_text() -> None:
    events = _session(_branch_records(), "f")
    assert [event.event_id for event in events] == ["a", "c", "d", "e", "f"]
    assert [event.content for event in events] == [
        "user: alpha",
        "compaction: retained summary",
        "branch_summary: branch summary",
        'assistant: [tool_call read call-1] {"path":"a.txt"}',
        "toolResult: [tool_result read call-1] file contents",
    ]
    assert _session(_branch_records(), "b")[1].content == "assistant: other"
    assert _session(_branch_records()) == events


def test_control_nodes_keep_parent_links_but_produce_no_event() -> None:
    header = {"type": "session", "version": 3, "id": "formats"}
    records = [
        {"type": "message", "id": "a", "parentId": None, "message": {"role": "user", "content": "Request"}},
        {"type": "thinking_level_change", "id": "b", "parentId": "a"},
        {
            "type": "message",
            "id": "c",
            "parentId": "b",
            "message": {"role": "bashExecution", "command": "printf result", "output": "result", "content": []},
        },
        {
            "type": "custom_message",
            "id": "d",
            "parentId": "c",
            "content": [{"type": "text", "text": "Custom evidence"}],
        },
        {
            "type": "message",
            "id": "e",
            "parentId": "d",
            "message": {
                "role": "assistant",
                "content": [{"type": "image", "data": "ignored"}, {"type": "text", "text": "Answer"}],
            },
        },
    ]
    raw = b"\xef\xbb\xbf" + "\n".join(json.dumps(row) for row in [header, *records]).encode()
    events = parse_pi_session(PiSessionInput(content=raw))
    assert [event.event_id for event in events] == ["a", "c", "d", "e"]
    assert [event.content for event in events] == [
        "user: Request",
        "bashExecution: $ printf result\nresult",
        "custom: Custom evidence",
        "assistant: Answer",
    ]
    assert events[1].parent_id == "b"
    assert events[1].record == records[2]


def test_empty_and_system_messages_have_no_visible_text() -> None:
    records = [
        {"type": "message", "id": "a", "parentId": None, "message": {"role": "system", "content": ""}},
        {"type": "message", "id": "b", "parentId": "a", "message": {"role": "user", "content": "hello"}},
        {"type": "usage", "id": "c", "parentId": "b", "kind": "cache_warm"},
        {"type": "message", "id": "d", "parentId": "c", "message": {"role": "assistant", "content": []}},
    ]
    events = _session(records)
    assert [event.event_id for event in events] == ["b"]


@pytest.mark.parametrize(
    "records,leaf",
    [
        ([{"type": "model_change", "id": "a", "parentId": "missing"}], "a"),
        ([{"type": "model_change", "id": "a", "parentId": "a"}], "a"),
        ([{"type": "model_change", "id": "a", "parentId": None}] * 2, "a"),
        ([{"type": "model_change", "id": "a", "parentId": None}], "absent"),
    ],
)
def test_invalid_graphs_fail_atomically(records, leaf) -> None:
    with pytest.raises(ValueError):
        _session(records, leaf)


@pytest.mark.parametrize(
    "raw", [b"", b"{}\n", b"[]\n", b"null\n", b"\xff", b'{"type":"session","version":2,"id":"x"}\n']
)
def test_invalid_archives_fail(raw) -> None:
    with pytest.raises(ValueError):
        parse_pi_session(PiSessionInput(content=raw))


def test_identity_is_stable_and_collision_free() -> None:
    assert pi_event_source_id("s", "e") == "pi:s:e"
    assert pi_event_source_id("a:b", "c") != pi_event_source_id("a", "b:c")
    assert pi_event_source_id("s", "e") == pi_event_source_id("s", "e")
    with pytest.raises(ValueError):
        pi_event_source_id("s" * 200, "e" * 200)


@_async_test
async def test_adapter_round_trip_and_projection() -> None:
    registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
    event = _session(_branch_records(), "f")[3]
    source = await registry.resolve(event)
    assert source.name == pi_event_source_id("session-1", "e")
    assert await registry.read(source) == event
    observation = project_source_for_transport(registry, source)
    evidence = cast("dict[str, JsonValue]", observation.projection(TEXT_EVIDENCE_PROJECTION_KEY))
    assert evidence["source_type"] == "pi-event"
    assert evidence["source_id"] == source.name
    assert evidence["content"] == event.content
    assert evidence["metadata"] == {
        "session_id": "session-1",
        "event_id": "e",
        "parent_id": "d",
        "event_type": "message",
        "role": "assistant",
    }


class _MemorySink:
    def __init__(self) -> None:
        self.registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
        self.submitted: dict[str, PiEventInput] = {}

    async def submit(self, binding, item_id, definition_name, value) -> SourceRef:
        source = await self.registry.resolve(value)
        self.submitted[source.name] = value
        return SourceRef(source_type=definition_name, source_id=source.name)


class _MemoryCheckpoints:
    def __init__(self) -> None:
        self.stored: JsonValue | None = None

    async def load(self, binding) -> JsonValue | None:
        return self.stored

    async def save(self, binding, checkpoint, *, expected) -> None:
        assert self.stored == expected
        self.stored = checkpoint


def _write(path: Path, records) -> None:
    header = {"type": "session", "version": 3, "id": "tree"}
    path.write_text("\n".join(json.dumps(row) for row in [header, *records]) + "\n", encoding="utf-8")


def _binding() -> ConnectorBinding:
    return ConnectorBinding(
        scope_id="tree", binding_id="pi", connector_name="pi-session-history", connector_version="1"
    )


@_async_test
async def test_connector_branch_switch_append_and_mutation(tmp_path: Path) -> None:
    path = tmp_path / "tree.jsonl"
    records = [
        {"type": "message", "id": "root", "parentId": None, "message": {"role": "user", "content": "Shared request"}},
        {
            "type": "message",
            "id": "left",
            "parentId": "root",
            "message": {"role": "assistant", "content": "Left evidence"},
        },
        {"type": "model_change", "id": "control", "parentId": "root"},
        {
            "type": "message",
            "id": "right",
            "parentId": "control",
            "message": {"role": "assistant", "content": "Right evidence"},
        },
    ]
    _write(path, records)
    sink = _MemorySink()
    checkpoints = _MemoryCheckpoints()
    lifecycle = ConnectorLifecycle(sink=sink, checkpoints=checkpoints)

    first = await lifecycle.run(PiSessionConnector(path, "left"), _binding())
    assert [item.item_id for item in first.items] == ["root", "left"]
    assert first.committed_checkpoint is not None
    assert first.previous_checkpoint is None

    second = await lifecycle.run(PiSessionConnector(path, "right"), _binding())
    assert [item.item_id for item in second.items] == ["right"]
    assert second.previous_checkpoint == first.committed_checkpoint

    replay = await lifecycle.run(PiSessionConnector(path, "right"), _binding())
    assert replay.items == ()
    assert replay.committed_checkpoint == second.committed_checkpoint

    records.append({
        "type": "message",
        "id": "tail",
        "parentId": "right",
        "message": {"role": "assistant", "content": "Later evidence"},
    })
    _write(path, records)
    added = await lifecycle.run(PiSessionConnector(path, "tail"), _binding())
    assert [item.item_id for item in added.items] == ["tail"]

    path.write_text(path.read_text() + "{bad json}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        await lifecycle.run(PiSessionConnector(path, "tail"), _binding())
    _write(path, records)
    repaired = await lifecycle.run(PiSessionConnector(path, "tail"), _binding())
    assert repaired.previous_checkpoint == added.committed_checkpoint
    assert repaired.items == ()

    records[0]["message"]["content"] = "Changed request"
    _write(path, records)
    with pytest.raises(ValueError):
        await lifecycle.run(PiSessionConnector(path, "tail"), _binding())
    records[0]["message"]["content"] = "Shared request"
    _write(path, records)
    stable = await lifecycle.run(PiSessionConnector(path, "tail"), _binding())
    assert stable.previous_checkpoint == added.committed_checkpoint
    assert stable.items == ()
