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

"""Durable Source lifecycle graphs with optimistic comparison.

One row per scope holds the whole dependency graph: every registered node,
its reference, parents, policy, and the last applied event per Source. The
graph is tiny (sixteen to twenty nodes per scope in the acceptance shape),
so storing it as one canonical payload keeps registration, event, and
availability reads and writes atomic without a table per node.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue
from sqlalchemy import insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.builtin.persistence.codec import dump_model, load_model, stored_bytes
from powercontext.builtin.persistence.errors import StoredPayloadConflictError
from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_STATES_TABLE

NodeFamily = Literal["source", "memory", "topic"]
DependencyPolicy = Literal["all", "any"]


class SourceLifecycleNode(BaseModel):
    """One registered node in a scope's Source lifecycle graph."""

    model_config = ConfigDict(frozen=True)

    node_id: str
    family: NodeFamily
    # The node's original reference (SourceRef, MemoryCitation, or
    # ArtifactRef) as JSON, so resolve() returns the exact reference it
    # was registered with.
    reference: JsonValue
    parents: list[str] = []
    policy: DependencyPolicy | None = None
    # Sources only: the lifecycle revision events must match exactly.
    revision: int | None = None
    available: bool = True
    # Sources only: the last applied event sequence and its full content,
    # used to accept replays and reject stale or conflicting events.
    last_sequence: int | None = None
    last_event: JsonValue | None = None


class SourceLifecycleState(BaseModel):
    """The whole lifecycle graph of one scope, in registration order.

    Registration order is a topological order: parents must already be
    registered, so availability can be recomputed by one forward pass.
    """

    model_config = ConfigDict(frozen=True)

    version: int = 1
    nodes: list[SourceLifecycleNode] = []


def derive_availability(nodes: Sequence[SourceLifecycleNode]) -> dict[str, bool]:
    """Compute every node's availability in one forward pass.

    Registration order is a topological order, so parents are always
    resolved before their children.
    """

    available: dict[str, bool] = {}
    for node in nodes:
        if node.family == "source":
            available[node.node_id] = node.available
        elif node.policy == "all":
            available[node.node_id] = all(available[parent] for parent in node.parents)
        else:
            available[node.node_id] = any(available[parent] for parent in node.parents)
    return available


async def unavailable_topic_refs(connection: AsyncConnection, /) -> set[tuple[str, str]]:
    """Return (scope_id, artifact_id) for Topic nodes the lifecycle graph marks unavailable.

    Topic Memory storage keeps every published head active; a lifecycle
    withdrawal is the one deliberate exception, so startup validation
    exempts exactly these heads instead of treating them as corruption.
    """

    rows = (await connection.execute(select(SOURCE_LIFECYCLE_STATES_TABLE))).mappings()
    unavailable: set[tuple[str, str]] = set()
    for row in rows:
        scope_id = str(row["scope_id"])
        state = load_model(
            SourceLifecycleState,
            stored_bytes(row["state"], column="state"),
            kind="source-lifecycle-state",
            name=scope_id,
        )
        available = derive_availability(state.nodes)
        for node in state.nodes:
            if node.family != "topic" or available[node.node_id]:
                continue
            reference = node.reference
            if isinstance(reference, dict) and isinstance(reference.get("artifact_id"), str):
                unavailable.add((scope_id, reference["artifact_id"]))
    return unavailable


class SourceLifecycleRepository:
    """Persist one scope's lifecycle graph with value-based comparison."""

    async def load(
        self,
        connection: AsyncConnection,
        scope_id: str,
        /,
        *,
        for_update: bool = False,
    ) -> SourceLifecycleState | None:
        row = await self._find_row(connection, scope_id, for_update=for_update)
        if row is None:
            return None
        return load_model(
            SourceLifecycleState,
            stored_bytes(row["state"], column="state"),
            kind="source-lifecycle-state",
            name=scope_id,
        )

    async def _find_row(
        self,
        connection: AsyncConnection,
        scope_id: str,
        *,
        for_update: bool,
    ):
        statement = select(SOURCE_LIFECYCLE_STATES_TABLE).where(
            SOURCE_LIFECYCLE_STATES_TABLE.c.scope_id == scope_id,
        )
        if for_update:
            statement = statement.with_for_update()
        return (await connection.execute(statement)).mappings().one_or_none()

    async def save(
        self,
        connection: AsyncConnection,
        scope_id: str,
        state: SourceLifecycleState,
        /,
        *,
        expected: SourceLifecycleState | None,
    ) -> None:
        existing_row = await self._find_row(connection, scope_id, for_update=True)
        actual = None if existing_row is None else self._decode_row(existing_row, scope_id)
        if actual != expected:
            raise StoredPayloadConflictError("source-lifecycle-state", scope_id)

        payload = dump_model(state, kind="source-lifecycle-state", name=scope_id)
        if existing_row is None:
            statement = insert(SOURCE_LIFECYCLE_STATES_TABLE).values(scope_id=scope_id, state=payload)
            try:
                if connection.dialect.name == "sqlite":
                    async with connection.begin_nested():
                        await connection.execute(statement)
                elif connection.dialect.name == "mysql":
                    await connection.execute(statement)
                else:
                    raise StoredPayloadConflictError("source-lifecycle-state", scope_id)
            except IntegrityError:
                # SQLite needs the nested transaction to keep the outer CAS
                # transaction usable after an insert race. OceanBase is
                # MySQL-compatible but may discard a write-path SAVEPOINT
                # before SQLAlchemy releases it, so execute directly there.
                raise StoredPayloadConflictError("source-lifecycle-state", scope_id) from None
        else:
            # Reaching here means expected is not None: an existing row with
            # expected=None would already have raised the conflict above.
            if expected is None:
                raise StoredPayloadConflictError("source-lifecycle-state", scope_id)
            result = await connection.execute(
                update(SOURCE_LIFECYCLE_STATES_TABLE)
                .where(
                    SOURCE_LIFECYCLE_STATES_TABLE.c.scope_id == scope_id,
                    SOURCE_LIFECYCLE_STATES_TABLE.c.state
                    == dump_model(expected, kind="source-lifecycle-state", name=scope_id),
                )
                .values(state=payload)
            )
            if result.rowcount != 1:
                raise StoredPayloadConflictError("source-lifecycle-state", scope_id)

    @staticmethod
    def _decode_row(row, scope_id: str) -> SourceLifecycleState:
        return load_model(
            SourceLifecycleState,
            stored_bytes(row["state"], column="state"),
            kind="source-lifecycle-state",
            name=scope_id,
        )


__all__ = [
    "DependencyPolicy",
    "NodeFamily",
    "SourceLifecycleNode",
    "SourceLifecycleRepository",
    "SourceLifecycleState",
    "derive_availability",
    "unavailable_topic_refs",
]
