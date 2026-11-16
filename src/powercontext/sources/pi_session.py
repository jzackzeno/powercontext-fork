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

"""Pi v3 session history: one captured Source per visible event.

A Pi session file is JSONL whose first line is a ``type=session`` header
(``version=3``) and whose remaining records form a tree through
``id``/``parentId`` links. The selected leaf and its ancestors define one
branch. Every visible event on that branch becomes an independent
``CAPTURED`` :class:`PiEventSource` that keeps the complete raw record and
the full event text, so citations read back exactly what was captured.
Control records (model changes, usage, labels and the like) keep the graph
connected but produce no Source. The connector imports only events that are
not yet in its persisted checkpoint and refuses to advance when an already
imported event was rewritten.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import quote

from pydantic import BaseModel, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    AdapterSourceDefinition,
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
    SourceDefinition,
    TextEvidence,
)
from powercontext.sources.models import Source, SourceMaterialization

PI_EVENT_SOURCE_NAME = "pi-event"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"
_PI_EVENT_SOURCE_ID_PREFIX = "pi:"
_CHECKPOINT_VERSION = 1


class PiSessionInput(BaseModel):
    """A whole Pi v3 session file plus the leaf selecting one branch."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event on the selected branch, with its raw record."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """A captured Pi session event that can be used as Artifact evidence."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Build the stable identity for one Pi session event.

    Both components are percent-encoded separately with
    ``urllib.parse.quote(component, safe="")`` so neither can collide with
    the ``:`` separators. The identity must not exceed 256 characters.
    """

    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")  # noqa: TRY003
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("event_id must be a non-empty string")  # noqa: TRY003
    identity = f"{_PI_EVENT_SOURCE_ID_PREFIX}{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"pi event source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return identity


def _decode_session_text(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session content must be valid UTF-8") from error  # noqa: TRY003


def _parse_record(line: str, position: int) -> dict[str, JsonValue]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"Pi session line {position} is not valid JSON") from error  # noqa: TRY003
    if not isinstance(value, dict):
        raise ValueError(f"Pi session line {position} must be a JSON object")  # noqa: TRY003, TRY004
    return value


def _read_session_records(text: str) -> tuple[str, list[dict[str, JsonValue]]]:
    """Validate the header and the whole record graph; return session id and records.

    Blank lines are ignored. Every record after the header must carry a
    unique non-empty string ``id`` and a ``parentId`` that is either null or
    the id of a record that already appeared, so a parent always precedes
    its children and cycles cannot form.
    """

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Pi session content must not be empty")  # noqa: TRY003
    header = _parse_record(lines[0], 1)
    if header.get("type") != "session" or header.get("version") != 3:
        raise ValueError("first Pi session record must be a session header with version 3")  # noqa: TRY003
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Pi session header id must be a non-empty string")  # noqa: TRY003
    by_id: dict[str, dict[str, JsonValue]] = {}
    records: list[dict[str, JsonValue]] = []
    for position, line in enumerate(lines[1:], start=2):
        record = _parse_record(line, position)
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"Pi session record at line {position} must have a non-empty string id")  # noqa: TRY003
        if record_id in by_id:
            raise ValueError(f"Pi session record id {record_id!r} is duplicated")  # noqa: TRY003
        record_type = record.get("type")
        if not isinstance(record_type, str) or not record_type:
            raise ValueError(f"Pi session record {record_id!r} must have a non-empty string type")  # noqa: TRY003
        if "parentId" not in record:
            raise ValueError(f"Pi session record {record_id!r} is missing parentId")  # noqa: TRY003
        parent_id = record["parentId"]
        if parent_id is not None and (not isinstance(parent_id, str) or parent_id not in by_id):
            raise ValueError(f"Pi session record {record_id!r} has a missing parent {parent_id!r}")  # noqa: TRY003
        by_id[record_id] = record
        records.append(record)
    return session_id, records


def _compact_json(value: object) -> str:
    """Canonical compact JSON: sorted keys, no extra spaces, Unicode kept."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _visible_text_parts(content: object) -> list[str]:
    """Visible text pieces of a message or custom content value.

    Plain strings are kept as-is; block lists contribute their text blocks
    and toolCall blocks, joined by newlines by the caller. Thinking, image
    and unknown blocks carry no visible text.
    """

    if isinstance(content, str):
        return [content] if content else []
    if not isinstance(content, list):
        return []
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        elif block_type == "toolCall":
            name = block.get("name") if isinstance(block.get("name"), str) else ""
            call_id = block.get("id") if isinstance(block.get("id"), str) else ""
            arguments = block.get("arguments", {})
            parts.append(f"[tool_call {name} {call_id}] {_compact_json(arguments)}")
    return parts


def _message_body(message: dict[str, JsonValue]) -> tuple[str, str] | None:
    """Return ``(role, body)`` for one message object, or ``None`` when malformed."""

    role = message.get("role")
    if not isinstance(role, str) or not role:
        return None
    if role == "toolResult":
        name = message.get("toolName") if isinstance(message.get("toolName"), str) else ""
        call_id = message.get("toolCallId") if isinstance(message.get("toolCallId"), str) else ""
        text = "\n".join(_visible_text_parts(message.get("content")))
        prefix = f"[tool_result {name} {call_id}]"
        return role, f"{prefix} {text}" if text else prefix
    if role == "bashExecution":
        raw_command = message.get("command")
        raw_output = message.get("output")
        command = raw_command if isinstance(raw_command, str) else ""
        output = raw_output if isinstance(raw_output, str) else ""
        parts: list[str] = []
        if command:
            parts.append(f"$ {command}")
        if output:
            parts.append(output)
        if not parts:
            parts = _visible_text_parts(message.get("content"))
        return role, "\n".join(parts)
    return role, "\n".join(_visible_text_parts(message.get("content")))


def _event_body(record: dict[str, JsonValue]) -> tuple[str, str] | None:
    """Return ``(role, body)`` for a visible event, or ``None`` for control records."""

    record_type = record.get("type")
    if record_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return None
        return _message_body(message)
    if record_type in {"compaction", "branch_summary"}:
        summary = record.get("summary")
        if not isinstance(summary, str) or not summary:
            return None
        return str(record_type), summary
    if record_type == "custom_message":
        return "custom", "\n".join(_visible_text_parts(record.get("content")))
    return None


def _visible_event(session_id: str, record: dict[str, JsonValue]) -> PiEventInput | None:
    """Build the event for one record, or ``None`` when it has no visible text."""

    resolved = _event_body(record)
    if resolved is None:
        return None
    role, body = resolved
    if not body:
        return None
    event_id = record.get("id")
    parent_id = record.get("parentId")
    event_type = record.get("type")
    if not isinstance(event_id, str) or not isinstance(event_type, str):
        raise ValueError("Pi session record must have string id and type")  # noqa: TRY003, TRY004
    if parent_id is not None and not isinstance(parent_id, str):
        raise ValueError("Pi session record parentId must be a string or null")  # noqa: TRY003
    # Also enforces the 256-character identity limit for this event.
    pi_event_source_id(session_id, event_id)
    return PiEventInput(
        session_id=session_id,
        event_id=event_id,
        parent_id=parent_id,
        event_type=event_type,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _all_visible_events(session_id: str, records: list[dict[str, JsonValue]]) -> tuple[PiEventInput, ...]:
    return tuple(event for record in records if (event := _visible_event(session_id, record)) is not None)


def _parse_session(value: PiSessionInput) -> tuple[str, list[dict[str, JsonValue]], tuple[PiEventInput, ...]]:
    """Parse and validate a whole session, then select the branch events."""

    session_id, records = _read_session_records(_decode_session_text(value.content))
    by_id = {str(record["id"]): record for record in records}
    leaf_id = value.leaf_id
    if leaf_id is None:
        if not records:
            raise ValueError("Pi session has no entries to select as leaf")  # noqa: TRY003
        leaf_id = str(records[-1]["id"])
    elif leaf_id not in by_id:
        raise ValueError(f"Pi session leaf {leaf_id!r} does not exist")  # noqa: TRY003
    chain: list[dict[str, JsonValue]] = []
    seen: set[str] = set()
    current: str | None = leaf_id
    while current is not None:
        if current in seen:
            raise ValueError(f"Pi session entries contain a cycle at {current!r}")  # noqa: TRY003
        seen.add(current)
        record = by_id[current]
        chain.append(record)
        parent = record.get("parentId")
        current = parent if isinstance(parent, str) else None
    chain.reverse()
    selected = tuple(event for record in chain if (event := _visible_event(session_id, record)) is not None)
    return session_id, records, selected


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Parse a Pi v3 session file and return the selected branch's visible events.

    Events run from the root to the leaf. The whole input is parsed and
    validated before anything is returned, so any error fails atomically
    with :class:`ValueError` and no partial events are produced.
    """

    return _parse_session(value)[2]


def _record_digest(record: dict[str, JsonValue]) -> str:
    return hashlib.sha256(_compact_json(record).encode("utf-8")).hexdigest()


def _load_checkpoint(checkpoint: JsonValue | None) -> dict[str, str]:
    """Read the persisted map of imported event identity to record digest."""

    if checkpoint is None:
        return {}
    if not isinstance(checkpoint, dict):
        raise ValueError("Pi session checkpoint must be a JSON object")  # noqa: TRY003, TRY004
    if checkpoint.get("version") != _CHECKPOINT_VERSION:
        raise ValueError("Pi session checkpoint version is not supported")  # noqa: TRY003
    events = checkpoint.get("events")
    if not isinstance(events, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in events.items()
    ):
        raise ValueError("Pi session checkpoint events must map identities to record digests")  # noqa: TRY003
    return dict(events)


class PiEventSourceAdapter:
    """Materialize one parsed Pi event as a captured Source."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        identity = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=identity,
            source_id=identity,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Expose one captured Pi event as standard text evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.name,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_ADAPTER = PiEventSourceAdapter()
PI_EVENT_SOURCE_DEFINITION: SourceDefinition[PiEventInput, PiEventSource, PiEventInput] = AdapterSourceDefinition(
    PI_EVENT_SOURCE_ADAPTER,
    projections=(PiEventTextEvidenceProjection(),),
)


class PiSessionConnector:
    """Import newly visible Pi session events for one branch at a time.

    Each run fully parses the file and the selected branch first, then
    checks every visible event in the file against the persisted
    checkpoint: an already imported event whose raw record changed is a
    rewrite and fails the run with :class:`ValueError` before anything is
    submitted. Only selected-branch events missing from the checkpoint are
    submitted, so shared ancestors are never duplicated and switching
    branches imports just the new events. The checkpoint is committed by
    the surrounding :class:`ConnectorLifecycle` only after the run
    completes, so a failed run keeps the previous progress.
    """

    name = PI_SESSION_CONNECTOR_NAME
    version = "1"
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        content = self.path.read_bytes()
        parsed = PiSessionInput(content=content, leaf_id=self.leaf_id)
        session_id, records, selected = _parse_session(parsed)
        imported = _load_checkpoint(session.checkpoint)
        digests = {
            pi_event_source_id(event.session_id, event.event_id): _record_digest(event.record)
            for event in _all_visible_events(session_id, records)
        }
        for identity, digest in digests.items():
            if identity in imported and imported[identity] != digest:
                raise ValueError(f"imported Pi event {identity!r} was rewritten")  # noqa: TRY003
        checkpoint_events = dict(imported)
        for event in selected:
            identity = pi_event_source_id(event.session_id, event.event_id)
            if identity in imported:
                continue
            await session.submit(event.event_id, PI_EVENT_SOURCE_NAME, event)
            checkpoint_events[identity] = digests[identity]
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={"version": _CHECKPOINT_VERSION, "events": checkpoint_events},
        )


__all__ = [
    "PI_EVENT_SOURCE_ADAPTER",
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_SESSION_CONNECTOR_NAME",
    "PiEventInput",
    "PiEventSource",
    "PiEventSourceAdapter",
    "PiEventTextEvidenceProjection",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
