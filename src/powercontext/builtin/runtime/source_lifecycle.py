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

"""Propagate Source availability through a dependency DAG.

A scope registers Sources, Memory entries, and Topic Memory artifacts as
nodes of a directed acyclic graph. Each derived node carries an ``all`` or
``any`` policy over its parents: ``all`` needs every parent available,
``any`` needs one available parent. A lifecycle event withdraws or restores
one Source; the new availability propagates to every descendant, and the
native Memory and Topic Memory states follow: withdrawn entries leave
search and Context (their text stays readable through the original
citation), restored entries come back.

The graph, the per-Source event sequence, and every node's availability
are persisted as one payload per scope, so a reopened runtime resumes
exactly where the previous one stopped. Native state writes run before
the persisted event is recorded: if a native write fails mid-propagation,
the event stays unrecorded and a retry replays the same propagation,
which is idempotent per entry because the native services no-op when an
entry is already in its target state.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictBool, StrictInt, StrictStr, field_validator

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.errors import StoredPayloadConflictError
from powercontext.builtin.persistence.source_lifecycle import (
    DependencyPolicy,
    NodeFamily,
    SourceLifecycleNode,
    SourceLifecycleState,
    derive_availability,
)
from powercontext.builtin.sources import validate_scope_id
from powercontext.errors import ArtifactNotFoundError
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.artifacts.memory import MemoryService
    from powercontext.builtin.artifacts.memory.models import MemoryEntryVersion
    from powercontext.builtin.runtime.application import BuiltinRuntime


class SourceLifecycleError(ValueError):
    """A lifecycle registration, event, or resolution broke the contract."""


class SourceLifecycleEvent(BaseModel):
    """One Source availability change, versioned and sequenced per Source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: StrictStr
    revision: StrictInt = Field(gt=0)
    sequence: StrictInt = Field(gt=0)
    available: StrictBool
    reason: StrictStr

    @field_validator("node_id", "reason")
    @classmethod
    def _require_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty or blank")  # noqa: TRY003
        return value


# Availability is derived from the whole graph; the persistence layer owns
# the derivation so startup validation can share it.
_availability = derive_availability


def _require_node_id(node_id: object) -> str:
    if not isinstance(node_id, str) or not node_id.strip():
        raise SourceLifecycleError("node_id must be a non-empty string")  # noqa: TRY003
    return node_id


def _require_revision(revision: object) -> int:
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise SourceLifecycleError("revision must be a positive integer")  # noqa: TRY003
    return revision


def _require_parents(parents: object, node_id: str) -> tuple[str, ...]:
    if isinstance(parents, str) or not isinstance(parents, Sequence):
        raise SourceLifecycleError("parents must be a sequence of node ids")  # noqa: TRY003
    checked = tuple(parents)
    if not checked:
        raise SourceLifecycleError("a derived node needs at least one parent")  # noqa: TRY003
    for parent in checked:
        if not isinstance(parent, str) or not parent.strip():
            raise SourceLifecycleError("parents must be non-empty node ids")  # noqa: TRY003
    if len(set(checked)) != len(checked):
        raise SourceLifecycleError("parents must not repeat")  # noqa: TRY003
    if node_id in checked:
        raise SourceLifecycleError("a node cannot depend on itself")  # noqa: TRY003
    return cast(tuple[str, ...], checked)


def _require_policy(policy: object) -> DependencyPolicy:
    if policy == "all" or policy == "any":
        return cast(DependencyPolicy, policy)
    raise SourceLifecycleError("policy must be 'all' or 'any'")  # noqa: TRY003


class SourceLifecycleApplication:
    """Select the Source lifecycle service for one scope."""

    def __init__(self, runtime: BuiltinRuntime) -> None:
        self._runtime = runtime

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycleApplication:
        return ScopedSourceLifecycleApplication(self._runtime, scope_id)


class ScopedSourceLifecycleApplication:
    """Register a dependency graph and propagate Source events in one scope."""

    def __init__(self, runtime: BuiltinRuntime, scope_id: str) -> None:
        self._runtime = runtime
        self.scope_id = validate_scope_id(scope_id)

    async def register_source(
        self,
        node_id: str,
        source_ref: SourceRef,
        /,
        *,
        revision: int,
    ) -> None:
        """Register one captured Source as a graph root. Identical repeats are no-ops."""

        node_id = _require_node_id(node_id)
        revision = _require_revision(revision)
        if not isinstance(source_ref, SourceRef):
            raise SourceLifecycleError("source_ref must be a SourceRef")  # noqa: TRY003
        async with self._runtime._scope_operation(self.scope_id), self._runtime._locked(self.scope_id):
            state, expected = await self._load_state()
            if self._already_registered(
                state, node_id, "source", source_ref.model_dump(mode="json"), (), None, revision
            ):
                return
            await self._validate_source_ref(source_ref)
            node = SourceLifecycleNode(
                node_id=node_id,
                family="source",
                reference=source_ref.model_dump(mode="json"),
                revision=revision,
                available=True,
            )
            await self._save(state.model_copy(update={"nodes": [*state.nodes, node]}), expected)

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        /,
        *,
        parents: Sequence[str] = (),
        policy: str = "all",
    ) -> None:
        """Register one Memory entry as a derived node.

        Its availability is computed from its parents' current state right
        away; an entry registered under withdrawn evidence is deactivated
        immediately instead of waiting for the next event.
        """

        node_id = _require_node_id(node_id)
        parents = _require_parents(parents, node_id)
        policy = _require_policy(policy)
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("citation must be a MemoryCitation")  # noqa: TRY003
        async with self._runtime._context(self.scope_id) as context, self._runtime._locked(self.scope_id):
            state, expected = await self._load_state()
            if self._already_registered(
                state, node_id, "memory", citation.model_dump(mode="json"), parents, policy, None
            ):
                return
            self._check_parents(state, node_id, parents)
            await self._validate_citation(context.artifacts.memory, context.artifacts.memory_artifact_id, citation)
            node = SourceLifecycleNode(
                node_id=node_id,
                family="memory",
                reference=citation.model_dump(mode="json"),
                parents=list(parents),
                policy=policy,
            )
            new_state = state.model_copy(update={"nodes": [*state.nodes, node]})
            if not _availability(new_state.nodes)[node_id]:
                await self._set_native_states(
                    context.artifacts.memory,
                    context.artifacts.memory_artifact_id,
                    [(node, False)],
                    reason="register",
                )
            await self._save(new_state, expected)

    async def register_topic(
        self,
        node_id: str,
        artifact_ref: ArtifactRef,
        /,
        *,
        parents: Sequence[str] = (),
        policy: str = "all",
    ) -> None:
        """Register one Topic Memory artifact as a derived node.

        Like Memory entries, a topic registered under withdrawn evidence is
        deactivated immediately.
        """

        node_id = _require_node_id(node_id)
        parents = _require_parents(parents, node_id)
        policy = _require_policy(policy)
        if not isinstance(artifact_ref, ArtifactRef):
            raise SourceLifecycleError("artifact_ref must be an ArtifactRef")  # noqa: TRY003
        async with self._runtime._scope_operation(self.scope_id), self._runtime._locked(self.scope_id):
            state, expected = await self._load_state()
            if self._already_registered(
                state, node_id, "topic", artifact_ref.model_dump(mode="json"), parents, policy, None
            ):
                return
            self._check_parents(state, node_id, parents)
            await self._validate_topic_ref(artifact_ref)
            node = SourceLifecycleNode(
                node_id=node_id,
                family="topic",
                reference=artifact_ref.model_dump(mode="json"),
                parents=list(parents),
                policy=policy,
            )
            new_state = state.model_copy(update={"nodes": [*state.nodes, node]})
            if not _availability(new_state.nodes)[node_id]:
                await self._runtime._provider.deactivate_topic_memory(self.scope_id, artifact_ref)
            await self._save(new_state, expected)

    async def apply(self, event: SourceLifecycleEvent, /) -> list[str]:
        """Apply one Source event and return every currently available node id, sorted.

        Replaying an already applied event returns the current active list
        without touching native state; a lower sequence or the same sequence
        with different content is rejected. Native state writes run before
        the event is recorded, so a failure leaves the event unrecorded and
        a retry completes the propagation.
        """

        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("event must be a SourceLifecycleEvent")  # noqa: TRY003
        async with self._runtime._context(self.scope_id) as context, self._runtime._locked(self.scope_id):
            state, expected = await self._load_state()
            node = self._find_node(state, event.node_id)
            if node is None or node.family != "source":
                raise SourceLifecycleError(f"unknown source: {event.node_id!r}")  # noqa: TRY003
            if node.revision != event.revision:
                raise SourceLifecycleError(  # noqa: TRY003
                    f"revision mismatch for {event.node_id!r}: event has {event.revision}, "
                    f"registered revision is {node.revision}"
                )
            if node.last_sequence is not None:
                if event.sequence < node.last_sequence:
                    raise SourceLifecycleError(  # noqa: TRY003
                        f"sequence went backwards for {event.node_id!r}: {event.sequence} < {node.last_sequence}"
                    )
                if event.sequence == node.last_sequence:
                    if self._same_event_content(node.last_event, event):
                        return sorted(node_id for node_id, ok in _availability(state.nodes).items() if ok)
                    raise SourceLifecycleError(  # noqa: TRY003
                        f"sequence {event.sequence} for {event.node_id!r} was already used with different content"
                    )
            updated = node.model_copy(
                update={
                    "available": event.available,
                    "last_sequence": event.sequence,
                    "last_event": event.model_dump(mode="json"),
                }
            )
            new_nodes = [updated if item.node_id == node.node_id else item for item in state.nodes]
            new_state = state.model_copy(update={"nodes": new_nodes})
            old_availability = _availability(state.nodes)
            new_availability = _availability(new_state.nodes)
            changed = [
                (item, new_availability[item.node_id])
                for item in new_state.nodes
                if old_availability[item.node_id] != new_availability[item.node_id]
            ]
            await self._set_native_states(
                context.artifacts.memory,
                context.artifacts.memory_artifact_id,
                changed,
                reason=event.reason,
            )
            await self._save(new_state, expected)
            return sorted(node_id for node_id, ok in new_availability.items() if ok)

    async def resolve(
        self,
        node_id: str,
        /,
        *,
        purpose: str,
    ) -> SourceRef | MemoryCitation | ArtifactRef:
        """Return a node's original reference.

        ``generation`` refuses unavailable nodes; ``audit`` returns the
        original reference regardless, so withdrawn history stays readable.
        """

        node_id = _require_node_id(node_id)
        if purpose not in ("generation", "audit"):
            raise SourceLifecycleError(f"unknown purpose: {purpose!r}")  # noqa: TRY003
        state, _ = await self._load_state()
        node = self._find_node(state, node_id)
        if node is None:
            raise SourceLifecycleError(f"unknown node: {node_id!r}")  # noqa: TRY003
        # Availability is derived from the whole graph, not from the stored
        # flag: only Source nodes get their flag updated by apply().
        if purpose == "generation" and not _availability(state.nodes)[node_id]:
            raise SourceLifecycleError(f"node is unavailable for generation: {node_id!r}")  # noqa: TRY003
        if node.family == "source":
            return SourceRef.model_validate(node.reference)
        if node.family == "memory":
            return MemoryCitation.model_validate(node.reference)
        return ArtifactRef.model_validate(node.reference)

    async def _load_state(self) -> tuple[SourceLifecycleState, SourceLifecycleState | None]:
        stored = await self._runtime._provider.source_lifecycle_state(self.scope_id)
        if stored is None:
            return SourceLifecycleState(), None
        return stored, stored

    async def _save(self, state: SourceLifecycleState, expected: SourceLifecycleState | None) -> None:
        try:
            await self._runtime._provider.commit_source_lifecycle_state(self.scope_id, state, expected=expected)
        except StoredPayloadConflictError as error:
            raise SourceLifecycleError("lifecycle state changed during the operation") from error  # noqa: TRY003

    @staticmethod
    def _find_node(state: SourceLifecycleState, node_id: str) -> SourceLifecycleNode | None:
        return next((node for node in state.nodes if node.node_id == node_id), None)

    def _already_registered(
        self,
        state: SourceLifecycleState,
        node_id: str,
        family: NodeFamily,
        reference: JsonValue,
        parents: Sequence[str],
        policy: DependencyPolicy | None,
        revision: int | None,
    ) -> bool:
        """Return True for an identical repeat; raise when the binding differs."""

        existing = self._find_node(state, node_id)
        if existing is None:
            return False
        if (
            existing.family != family
            or existing.reference != reference
            or existing.parents != list(parents)
            or existing.policy != policy
            or existing.revision != revision
        ):
            raise SourceLifecycleError(f"node is already registered with a different binding: {node_id!r}")  # noqa: TRY003
        return True

    @staticmethod
    def _check_parents(state: SourceLifecycleState, node_id: str, parents: Sequence[str]) -> None:
        known = {node.node_id: node for node in state.nodes}
        for parent in parents:
            if parent not in known:
                raise SourceLifecycleError(f"parent is not registered: {parent!r}")  # noqa: TRY003
        # Parents must already be registered, so a cycle can only pass
        # through the new node; verify the reachable subgraph anyway.
        seen: set[str] = set()
        stack = list(parents)
        while stack:
            current = stack.pop()
            if current == node_id:
                raise SourceLifecycleError("dependency cycle")  # noqa: TRY003
            if current in seen:
                continue
            seen.add(current)
            stack.extend(known[current].parents)

    @staticmethod
    def _same_event_content(stored: JsonValue | None, event: SourceLifecycleEvent) -> bool:
        if not isinstance(stored, dict):
            return False
        previous = dict(stored)
        previous.pop("sequence", None)
        current = event.model_dump(mode="json")
        current.pop("sequence", None)
        return previous == current

    async def _validate_source_ref(self, source_ref: SourceRef) -> None:
        try:
            await self._runtime.records.for_scope(self.scope_id).get_source(
                source_ref.source_type, source_ref.source_id
            )
        except Exception as error:
            raise SourceLifecycleError(f"source is not readable: {source_ref!r}") from error  # noqa: TRY003

    async def _validate_citation(
        self,
        service: MemoryService,
        memory_artifact_id: str,
        citation: MemoryCitation,
    ) -> None:
        if citation.memory_ref.artifact_id != memory_artifact_id:
            raise SourceLifecycleError("citation belongs to a different memory")  # noqa: TRY003
        try:
            await service.validate_citation(citation)
        except Exception as error:
            raise SourceLifecycleError(f"citation is not readable: {citation!r}") from error  # noqa: TRY003

    async def _validate_topic_ref(self, artifact_ref: ArtifactRef) -> None:
        if artifact_ref.family != "topic-memory":
            raise SourceLifecycleError(f"artifact_ref must reference topic-memory, got {artifact_ref.family!r}")  # noqa: TRY003
        try:
            await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
                "topic-memory", artifact_ref.artifact_id, artifact_ref.revision
            )
        except Exception as error:
            raise SourceLifecycleError(f"topic is not readable: {artifact_ref!r}") from error  # noqa: TRY003

    async def _set_native_states(
        self,
        service: MemoryService,
        memory_artifact_id: str,
        changes: Sequence[tuple[SourceLifecycleNode, bool]],
        *,
        reason: str,
    ) -> None:
        """Apply native Memory and Topic Memory state changes for flipped nodes."""

        if not changes:
            return
        to_inactive: list[MemoryEntryVersion] = []
        to_active: list[MemoryEntryVersion] = []
        topics_off: list[ArtifactRef] = []
        topics_on: list[ArtifactRef] = []
        for node, available in changes:
            if node.family == "memory":
                citation = MemoryCitation.model_validate(node.reference)
                version = await self._memory_entry(service, memory_artifact_id, citation)
                (to_active if available else to_inactive).append(version)
            elif node.family == "topic":
                ref = ArtifactRef.model_validate(node.reference)
                (topics_on if available else topics_off).append(ref)
        if to_inactive or to_active:
            head = await self._memory_head(service, memory_artifact_id)
            if to_inactive:
                head = await service.forget(head, entries=tuple(to_inactive), reason=reason)
            if to_active:
                head = await service.reactivate(head, entries=tuple(to_active), reason=reason)
        for ref in topics_off:
            await self._runtime._provider.deactivate_topic_memory(self.scope_id, ref)
        for ref in topics_on:
            await self._runtime._provider.reactivate_topic_memory(self.scope_id, ref)

    @staticmethod
    async def _memory_head(service: MemoryService, memory_artifact_id: str):
        try:
            return await service.head(memory_artifact_id)
        except ArtifactNotFoundError as error:
            raise SourceLifecycleError("memory does not exist") from error  # noqa: TRY003

    async def _memory_entry(
        self,
        service: MemoryService,
        memory_artifact_id: str,
        citation: MemoryCitation,
    ) -> MemoryEntryVersion:
        head = await self._memory_head(service, memory_artifact_id)
        for version in await service.entries(head):
            if version.entry_id == citation.entry_id:
                return version
        raise SourceLifecycleError(f"memory entry is not readable: {citation.entry_id!r}")  # noqa: TRY003


__all__ = [
    "ScopedSourceLifecycleApplication",
    "SourceLifecycleApplication",
    "SourceLifecycleError",
    "SourceLifecycleEvent",
]
