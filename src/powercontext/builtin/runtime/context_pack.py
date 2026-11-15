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

"""Incremental context packs with prefix reuse.

A :class:`ContextPack` is a complete prepared context split into deterministic
byte chunks. A :class:`ContextPackDelta` lets a receiver that already holds a
base pack rebuild a target pack from a shared chunk prefix plus the appended
suffix, so repeated prepares only pay for what changed. Both objects are
self-describing on the wire: fixed JSON fields, standard Base64 chunks,
sorted keys, compact separators, and a SHA-256 digest over the unsigned pack.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_PACK_SCHEMA = "powercontext.context-pack.v1"
_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

#: Chunk size used when a scope prepares incrementally. Smaller than the
#: 512-byte ``from_content`` default so that a mid-content change wastes less
#: of the shared prefix: only the partial chunk at the divergence point is
#: resent, while per-chunk framing stays small relative to the content.
#: 128 bytes keeps a ~400-byte stable header prefix (three full chunks) in
#: the common case where the first item changes, without the framing cost
#: of much smaller chunks.
INCREMENTAL_CHUNK_BYTES = 128


class ContextPackError(ValueError):
    """A context-pack protocol or chunking violation."""


@dataclass(frozen=True, slots=True)
class ContextPack:
    """A complete prepared context as deterministic byte chunks."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple):
            object.__setattr__(self, "items", tuple(self.items))

    @classmethod
    def build(cls, items: tuple[bytes, ...] | list[bytes] | bytes) -> ContextPack:
        """Build a complete pack from raw chunks."""

        if isinstance(items, (bytes, bytearray, memoryview)):
            items = (bytes(items),)
        normalized = tuple(_require_chunk(item) for item in items)
        encoded = [base64.b64encode(item).decode("ascii") for item in normalized]
        digest = _pack_digest(encoded)
        byte_count = 0
        for _ in range(16):
            payload = {"schema": _PACK_SCHEMA, "items": encoded, "digest": digest, "byte_count": byte_count}
            serialized = _canonical_json(payload)
            if len(serialized) == byte_count:
                return cls(items=normalized, digest=digest, byte_count=byte_count)
            byte_count = len(serialized)
        raise ContextPackError("cannot determine pack byte count")  # noqa: TRY003

    @classmethod
    def from_content(cls, content: str | bytes, *, chunk_bytes: int = 512) -> ContextPack:
        """Split content into valid UTF-8 chunks of at most ``chunk_bytes``."""

        if type(chunk_bytes) is not int or chunk_bytes < 1:
            raise ContextPackError("chunk_bytes must be a positive integer")  # noqa: TRY003
        if isinstance(content, str):
            data = content.encode("utf-8")
        elif isinstance(content, (bytes, bytearray, memoryview)):
            data = bytes(content)
            try:
                data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ContextPackError("content must be valid UTF-8") from exc  # noqa: TRY003
        else:
            raise ContextPackError("content must be text or bytes")  # noqa: TRY003
        return cls.build(_split_utf8(data, chunk_bytes))

    def to_bytes(self) -> bytes:
        """Serialize the complete pack to its canonical wire form."""

        payload = {
            "schema": _PACK_SCHEMA,
            "items": [base64.b64encode(item).decode("ascii") for item in self.items],
            "digest": self.digest,
            "byte_count": self.byte_count,
        }
        return _canonical_json(payload)

    @classmethod
    def from_bytes(cls, raw: bytes | bytearray | memoryview) -> ContextPack:
        """Decode and verify a serialized complete pack."""

        payload = _decode_json(raw, {"schema", "items", "digest", "byte_count"})
        if payload["schema"] != _PACK_SCHEMA:
            raise ContextPackError("unexpected pack schema")  # noqa: TRY003
        raw_items = payload["items"]
        if not isinstance(raw_items, list):
            raise ContextPackError("pack items must be a list")  # noqa: TRY003
        encoded_items: list[str] = []
        for value in raw_items:
            if not isinstance(value, str):
                raise ContextPackError("pack chunks must be Base64 strings")  # noqa: TRY003
            encoded_items.append(value)
        items = tuple(_decode_chunk(value) for value in encoded_items)
        digest = payload["digest"]
        if not isinstance(digest, str) or digest != _pack_digest(encoded_items):
            raise ContextPackError("pack digest mismatch")  # noqa: TRY003
        byte_count = payload["byte_count"]
        if type(byte_count) is not int or byte_count != len(bytes(raw)):
            raise ContextPackError("pack byte_count mismatch")  # noqa: TRY003
        return cls(items=items, digest=digest, byte_count=byte_count)


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """A prefix-preserving delta from one pack to another."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.appended, tuple):
            object.__setattr__(self, "appended", tuple(self.appended))

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        """Select the longest shared chunk prefix and carry the target suffix."""

        if not isinstance(base, ContextPack) or not isinstance(target, ContextPack):
            raise ContextPackError("delta endpoints must be context packs")  # noqa: TRY003
        prefix = 0
        for base_item, target_item in zip(base.items, target.items, strict=False):
            if base_item != target_item:
                break
            prefix += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix,
            appended=target.items[prefix:],
        )

    @property
    def byte_count(self) -> int:
        """Serialized delta size; not an extra field in the delta JSON."""

        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Serialize the delta to its canonical wire form."""

        payload = {
            "schema": _DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": [base64.b64encode(item).decode("ascii") for item in self.appended],
        }
        return _canonical_json(payload)

    @classmethod
    def from_bytes(cls, raw: bytes | bytearray | memoryview) -> ContextPackDelta:
        """Decode and verify a serialized delta."""

        payload = _decode_json(raw, {"schema", "base_digest", "target_digest", "prefix_items", "appended"})
        if payload["schema"] != _DELTA_SCHEMA:
            raise ContextPackError("unexpected delta schema")  # noqa: TRY003
        base_digest = payload["base_digest"]
        target_digest = payload["target_digest"]
        if (
            not isinstance(base_digest, str)
            or _DIGEST_PATTERN.match(base_digest) is None
            or not isinstance(target_digest, str)
            or _DIGEST_PATTERN.match(target_digest) is None
        ):
            raise ContextPackError("delta digests must be sha256 digests")  # noqa: TRY003
        prefix_items = payload["prefix_items"]
        if type(prefix_items) is not int or prefix_items < 0:
            raise ContextPackError("delta prefix_items must be a non-negative integer")  # noqa: TRY003
        raw_appended = payload["appended"]
        if not isinstance(raw_appended, list):
            raise ContextPackError("delta appended must be a list")  # noqa: TRY003
        appended = tuple(_decode_chunk(value) for value in raw_appended)
        return cls(
            base_digest=base_digest,
            target_digest=target_digest,
            prefix_items=prefix_items,
            appended=appended,
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Verify a delta against its base pack and return the target pack."""

    if not isinstance(base, ContextPack) or not isinstance(delta, ContextPackDelta):
        raise ContextPackError("delta application needs a pack and a delta")  # noqa: TRY003
    if delta.base_digest != base.digest:
        raise ContextPackError("delta base digest does not match base pack")  # noqa: TRY003
    if type(delta.prefix_items) is not int or delta.prefix_items < 0 or delta.prefix_items > len(base.items):
        raise ContextPackError("delta prefix_items is out of range")  # noqa: TRY003
    target = ContextPack.build(base.items[: delta.prefix_items] + delta.appended)
    if target.digest != delta.target_digest:
        raise ContextPackError("delta target digest mismatch")  # noqa: TRY003
    return target


def _pack_digest(encoded_items: Sequence[str]) -> str:
    unsigned = {"schema": _PACK_SCHEMA, "items": list(encoded_items)}
    return "sha256:" + hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _decode_json(raw: bytes | bytearray | memoryview, fields: set[str]) -> dict[str, object]:
    if isinstance(raw, (bytearray, memoryview)):
        raw = bytes(raw)
    if not isinstance(raw, bytes):
        raise ContextPackError("serialized pack must be bytes")  # noqa: TRY003
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise ContextPackError("serialized pack is not valid JSON") from exc  # noqa: TRY003
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ContextPackError("serialized pack must contain exactly the expected fields")  # noqa: TRY003
    return payload


def _decode_chunk(value: object) -> bytes:
    if not isinstance(value, str):
        raise ContextPackError("pack chunks must be Base64 strings")  # noqa: TRY003
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ContextPackError("pack chunk is not valid Base64") from exc  # noqa: TRY003


def _require_chunk(item: object) -> bytes:
    if not isinstance(item, (bytes, bytearray, memoryview)):
        raise ContextPackError("pack items must be bytes")  # noqa: TRY003
    return bytes(item)


def _split_utf8(data: bytes, chunk_bytes: int) -> tuple[bytes, ...]:
    """Split bytes into chunks that never break a UTF-8 character."""

    chunks: list[bytes] = []
    start = 0
    size = len(data)
    while start < size:
        end = min(start + chunk_bytes, size)
        if end < size:
            cut = end
            while cut > start and (data[cut] & 0xC0) == 0x80:
                cut -= 1
            if cut == start:
                raise ContextPackError("a single character exceeds chunk_bytes")  # noqa: TRY003
            end = cut
        chunks.append(data[start:end])
        start = end
    return tuple(chunks)


__all__ = [
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
