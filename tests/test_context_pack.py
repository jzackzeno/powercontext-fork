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

"""Focused tests for incremental context packs."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import cast

import pytest

from powercontext.builtin.artifacts.memory import MemoryCandidateRequest, MemoryEntryInput
from powercontext.builtin.runtime import BuiltinConfig, PrepareContextRequest, open_builtin_runtime
from powercontext.builtin.runtime.context_pack import (
    ContextPack,
    ContextPackDelta,
    ContextPackError,
    apply_context_pack_delta,
)
from powercontext.builtin.scope import ScopeDraft
from powercontext.builtin.sources import ContentSource


class _CandidatePipeline:
    async def extract(self, request: MemoryCandidateRequest, /) -> tuple[MemoryEntryInput, ...]:
        return tuple(
            MemoryEntryInput(kind="conversation-turn", text=source.content, sources=(source,))
            for source in request.sources
            if isinstance(source, ContentSource)
        )


def test_wire_payload_matches_oracle_for_other_items() -> None:
    items = (b"hello world", "你好世界".encode(), b"tail\n")
    unsigned = {
        "schema": "powercontext.context-pack.v1",
        "items": [base64.b64encode(item).decode("ascii") for item in items],
    }
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    pack = ContextPack.build(items)
    payload = json.loads(pack.to_bytes())
    assert payload == {
        **unsigned,
        "digest": "sha256:" + hashlib.sha256(canonical).hexdigest(),
        "byte_count": len(pack.to_bytes()),
    }
    assert pack.to_bytes() == json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert ContextPack.from_bytes(pack.to_bytes()) == pack


def test_byte_count_fixed_point_across_digit_boundaries() -> None:
    for total in (1, 9, 10, 50, 90, 99, 100, 500, 1000):
        pack = ContextPack.build((b"x" * total,))
        assert pack.byte_count == len(pack.to_bytes())
        assert ContextPack.from_bytes(pack.to_bytes()) == pack


@pytest.mark.parametrize("chunk_bytes", [0, -1, True, 1.5, "8"])
def test_from_content_rejects_bad_chunk_size(chunk_bytes: object) -> None:
    with pytest.raises(ContextPackError):
        ContextPack.from_content("abc", chunk_bytes=cast(int, chunk_bytes))


def test_from_content_never_splits_characters() -> None:
    content = "a海b洋c"
    for chunk_bytes in (3, 4, 5, 6, 7):
        pack = ContextPack.from_content(content, chunk_bytes=chunk_bytes)
        assert b"".join(pack.items).decode("utf-8") == content
        assert all(len(item) <= chunk_bytes for item in pack.items)
        assert all(item.decode("utf-8") for item in pack.items)
    for chunk_bytes in (1, 2):
        with pytest.raises(ContextPackError):
            ContextPack.from_content(content, chunk_bytes=chunk_bytes)
    with pytest.raises(ContextPackError):
        ContextPack.from_content("海", chunk_bytes=2)
    with pytest.raises(ContextPackError):
        ContextPack.from_content(b"\xff\xfe", chunk_bytes=8)


def test_delta_between_identical_packs_is_empty() -> None:
    pack = ContextPack.build((b"a", b"b"))
    delta = ContextPackDelta.between(pack, ContextPack.build((b"a", b"b")))
    assert delta.prefix_items == 2
    assert delta.appended == ()
    assert delta.base_digest == delta.target_digest == pack.digest
    assert apply_context_pack_delta(pack, delta) == pack


def test_delta_between_empty_and_nonempty() -> None:
    empty = ContextPack.build(())
    target = ContextPack.build((b"x",))
    delta = ContextPackDelta.between(empty, target)
    assert delta.prefix_items == 0
    assert delta.appended == (b"x",)
    assert apply_context_pack_delta(empty, ContextPackDelta.from_bytes(delta.to_bytes())) == target
    assert ContextPackDelta.between(target, empty).appended == ()
    assert apply_context_pack_delta(target, ContextPackDelta.between(target, empty)) == empty


def test_pack_decoders_reject_bad_fields_and_counts() -> None:
    pack = ContextPack.build((b"alpha", b"beta"))
    payload = json.loads(pack.to_bytes())
    for field in payload:
        incomplete = {key: value for key, value in payload.items() if key != field}
        with pytest.raises(ContextPackError):
            ContextPack.from_bytes(json.dumps(incomplete).encode())
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(json.dumps({**payload, "extra": 1}).encode())
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(json.dumps({**payload, "byte_count": payload["byte_count"] + 1}).encode())
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(json.dumps({**payload, "items": ["!!!not-base64!!!"]}).encode())
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(json.dumps({**payload, "items": [123]}).encode())
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(b"{not json")
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(json.dumps(["not", "a", "dict"]).encode())


def test_delta_decoders_reject_bad_digests_and_prefix() -> None:
    base = ContextPack.build((b"alpha",))
    delta = ContextPackDelta.between(base, ContextPack.build((b"alpha", b"beta")))
    payload = json.loads(delta.to_bytes())
    for bad in ("sha256:xyz", "sha256:" + "G" * 64, 123, None):
        with pytest.raises(ContextPackError):
            ContextPackDelta.from_bytes(json.dumps({**payload, "target_digest": bad}).encode())
    for bad_prefix in (True, 1.5, -1, "1"):
        with pytest.raises(ContextPackError):
            ContextPackDelta.from_bytes(json.dumps({**payload, "prefix_items": bad_prefix}).encode())
    with pytest.raises(ContextPackError):
        ContextPackDelta.from_bytes(json.dumps({**payload, "appended": ["!!!"]}).encode())
    with pytest.raises(ContextPackError):
        ContextPackDelta.from_bytes(json.dumps({**payload, "schema": "wrong"}).encode())


def test_apply_rejects_prefix_overflow_and_target_mismatch() -> None:
    base = ContextPack.build((b"a",))
    target = ContextPack.build((b"a", b"b"))
    delta = ContextPackDelta.between(base, target)
    with pytest.raises(ContextPackError):
        apply_context_pack_delta(base, ContextPackDelta(delta.base_digest, delta.target_digest, 2, ()))
    with pytest.raises(ContextPackError):
        apply_context_pack_delta(base, ContextPackDelta(delta.base_digest, "sha256:" + "0" * 64, 1, (b"b",)))
    with pytest.raises(ContextPackError):
        apply_context_pack_delta(base, ContextPackDelta("sha256:" + "0" * 64, delta.target_digest, 1, (b"b",)))


def test_direct_construction_normalizes_sequences() -> None:
    pack = ContextPack.build((b"a", b"b"))
    rebuilt = ContextPack(items=cast("tuple[bytes, ...]", [b"a", b"b"]), digest=pack.digest, byte_count=pack.byte_count)
    assert rebuilt == pack
    assert isinstance(rebuilt.items, tuple)
    delta = ContextPackDelta(pack.digest, pack.digest, 2, cast("tuple[bytes, ...]", [b"a", b"b"]))
    assert isinstance(delta.appended, tuple)
    assert delta.appended == (b"a", b"b")
    assert ContextPack.build(b"alpha") == ContextPack.build((b"alpha",))


@pytest.mark.asyncio
async def test_prepare_incremental_empty_and_unchanged_rounds() -> None:
    async with open_builtin_runtime(BuiltinConfig(), candidate_pipeline=_CandidatePipeline()) as runtime:
        scopes = runtime.scopes
        assert scopes is not None
        scope = await scopes.create(
            ScopeDraft(title="pack-empty", summary="pack-empty", idempotency_key="context-02-pack-empty")
        )
        context = runtime.context.for_scope(scope.scope_id)
        request = PrepareContextRequest(query="nothing here", max_bytes=4096)
        first = await context.prepare_incremental(request)
        assert isinstance(first, ContextPack)
        assert first.items == ()
        assert ContextPack.from_bytes(first.to_bytes()) == first
        second = await context.prepare_incremental(request, base_pack=first)
        assert isinstance(second, ContextPackDelta)
        assert second.prefix_items == 0
        assert second.appended == ()
        assert apply_context_pack_delta(first, second) == first
        with pytest.raises(ContextPackError):
            await context.prepare_incremental(request, base_pack=cast(ContextPack, "not-a-pack"))
        positional = await context.prepare_incremental(request, first)
        assert isinstance(positional, ContextPackDelta)
        assert apply_context_pack_delta(first, positional) == first
        keyword_request = await context.prepare_incremental(request=request, base_pack=first)
        assert isinstance(keyword_request, ContextPackDelta)

        async def _allow(_scopes: tuple[str, ...]) -> None:
            return None

        authorized = await context.prepare_incremental(request, authorize_scopes=_allow)
        assert isinstance(authorized, ContextPack)
        assert authorized == first
