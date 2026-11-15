"""Count-triggered Memory processing through public runtime APIs.

These tests cover the behaviours the implementation itself chooses: the
Memory loop that re-checks the remaining count after each window, and the
startup scan that re-admits work left pending by an earlier term. The public
acceptance suite covers the full matrix; these pin the two decisions a
reader of the driver most needs to trust.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections.abc import Callable, Coroutine
from contextlib import contextmanager
from functools import wraps
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ParamSpec, cast

import pytest
from pydantic import AnyHttpUrl, ValidationError

from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import BuiltinConfig, CaptureSource, open_builtin_runtime
from powercontext.builtin.runtime.config import InferenceConfig, RuntimeConfig
from powercontext.builtin.scope import ScopeDraft
from powercontext.builtin.triggers import SOURCE_WINDOW_TRIGGER_NAME

WAIT_SECONDS = 45
QUIET_SECONDS = 5

P = ParamSpec("P")


def _async_test(function: Callable[P, Coroutine[Any, Any, None]]) -> Callable[P, None]:
    @wraps(function)
    def run(*args: P.args, **kwargs: P.kwargs) -> None:
        asyncio.run(function(*args, **kwargs))

    return run


class _Handler(BaseHTTPRequestHandler):
    """Deterministic offline provider: echo the evidence it is given."""

    def log_message(self, format, *args):  # noqa: A002 - BaseHTTPRequestHandler signature
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/embeddings"):
            inputs = payload["input"]
            items = [inputs] if isinstance(inputs, str) else inputs
            response = {
                "object": "list",
                "data": [{"object": "embedding", "index": i, "embedding": [1.0, 0.0]} for i, _ in enumerate(items)],
                "model": "count-test-embedding",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
        else:
            prompts = [m["content"] for m in payload["messages"] if m["role"] == "user"]
            value = json.loads(prompts[-1])
            evidence = value.get("evidence", [])
            if "current_entries" in value:
                output = {
                    "candidates": [
                        {
                            "intent": "add",
                            "kind": "fact",
                            "text": self._text(item),
                            "evidence_ids": [item["evidence_id"]],
                        }
                        for item in evidence
                    ]
                }
            else:
                output = {
                    "probes": [
                        {"query": "CountEvidence", "keywords": [], "evidence_ids": [item["evidence_id"]]}
                        for item in evidence
                    ]
                }
            response = {
                "id": "offline-count",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": json.dumps(output)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }
        data = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    @staticmethod
    def _text(item):
        content = item["content"]
        text = content if isinstance(content, str) else json.dumps(content, sort_keys=True, ensure_ascii=False)
        return "CountEvidence " + hashlib.sha256(text.encode()).hexdigest()[:16] + " " + text[:600]


class _Server(ThreadingHTTPServer):
    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)


@contextmanager
def _provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-count-fixture")
    server = _Server()
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def _config(database: Path, server, **settings) -> BuiltinConfig:
    endpoint = AnyHttpUrl(f"http://127.0.0.1:{server.server_port}/v1")
    return BuiltinConfig(
        database=SQLiteConfig(url=f"sqlite+aiosqlite:///{database}"),
        inference=InferenceConfig(
            generation_model="openai-chat:gpt-4o-mini",
            generation_base_url=endpoint,
            embedding_model="openai:count-test-embedding",
            embedding_base_url=endpoint,
            embedding_dimension=2,
            embedding_profile_id="count-test",
        ),
        runtime=RuntimeConfig(
            artifact_processing_families=("memory",),
            artifact_processing_worker_timeout_seconds=30,
            memory_worker_timeout_seconds=30,
            **settings,
        ),
    )


def _sources(count: int) -> list[CaptureSource]:
    return [
        CaptureSource(
            source_id=f"count-source-{index}",
            content=f"Count fixture source {index} with enough text.",
            metadata={"role": "assistant"},
        )
        for index in range(count)
    ]


async def _scope(runtime, key="main"):
    return (await runtime.scopes.create(ScopeDraft(title=key, summary="count fixture", idempotency_key=key))).scope_id


async def _capture(runtime, scope_id, items):
    return [await runtime.sources.for_scope(scope_id).capture(item) for item in items]


async def _refs(runtime, scope_id):
    rows = (await runtime.memory.for_scope(scope_id).list()).entries
    return {(ref.source_type, ref.source_id) for row in rows for ref in row.entry.sources}


async def _wait_refs(runtime, scope_id, expected):
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        actual = await _refs(runtime, scope_id)
        if actual == expected:
            return
        assert actual <= expected, ("Unexpected consumed Source", actual, expected)
        await asyncio.sleep(0.15)
    assert await _refs(runtime, scope_id) == expected, "Count-triggered processing timed out"


async def _stay_refs(runtime, scope_id, expected):
    deadline = time.monotonic() + QUIET_SECONDS
    while time.monotonic() < deadline:
        assert await _refs(runtime, scope_id) == expected
        await asyncio.sleep(0.1)


def test_strict_optional_positive_thresholds():
    for field in ("memory_source_count_threshold", "topic_memory_source_count_threshold"):
        assert field in RuntimeConfig.model_fields
        assert getattr(RuntimeConfig(), field) is None
    for value in (None, 1, 3, 100):
        assert RuntimeConfig(memory_source_count_threshold=value).memory_source_count_threshold == value
        assert RuntimeConfig(topic_memory_source_count_threshold=value).topic_memory_source_count_threshold == value
    for value in (True, False, 0, -1, 1.0, "3"):
        with pytest.raises(ValidationError):
            RuntimeConfig(memory_source_count_threshold=cast(Any, value))
        with pytest.raises(ValidationError):
            RuntimeConfig(topic_memory_source_count_threshold=cast(Any, value))


@_async_test
async def test_pending_count_is_journal_identity_not_captures(tmp_path, monkeypatch):
    with _provider(monkeypatch) as server:
        settings = _config(tmp_path / "pending.db", server)
        async with open_builtin_runtime(settings) as runtime:
            sid = await _scope(runtime)
            provider = runtime._provider
            assert await provider.pending_source_count(sid, SOURCE_WINDOW_TRIGGER_NAME) == 0
            receipts = await _capture(runtime, sid, _sources(2))
            assert await provider.pending_source_count(sid, SOURCE_WINDOW_TRIGGER_NAME) == 2
            replay = await _capture(runtime, sid, _sources(1))
            assert replay[0].sequence == receipts[0].sequence
            assert await provider.pending_source_count(sid, SOURCE_WINDOW_TRIGGER_NAME) == 2
            await runtime.memory.for_scope(sid).flush()
            assert await provider.pending_source_count(sid, SOURCE_WINDOW_TRIGGER_NAME) == 0


@_async_test
async def test_memory_loop_consumes_while_remaining_meets_threshold(tmp_path, monkeypatch):
    """One batch of four with threshold two and window one consumes three, not one or all."""

    with _provider(monkeypatch) as server:
        settings = _config(tmp_path / "loop.db", server, memory_source_count_threshold=2, source_window_limit=1)
        async with open_builtin_runtime(settings) as runtime:
            sid = await _scope(runtime)
            receipts = await _capture(runtime, sid, _sources(4))
            expected = {(receipt.source_ref.source_type, receipt.source_ref.source_id) for receipt in receipts[:3]}
            await _wait_refs(runtime, sid, expected)
            await _stay_refs(runtime, sid, expected)
            assert (await runtime.memory.for_scope(sid).cursor()).sequence == receipts[2].sequence


@_async_test
async def test_startup_scan_admits_pending_without_new_input(tmp_path, monkeypatch):
    """Work captured while the threshold was off is admitted when a later term turns it on."""

    database = tmp_path / "scan.db"
    with _provider(monkeypatch) as server:
        async with open_builtin_runtime(_config(database, server)) as runtime:
            sid = await _scope(runtime)
            receipts = await _capture(runtime, sid, _sources(2))
            await _stay_refs(runtime, sid, set())
        async with open_builtin_runtime(_config(database, server, memory_source_count_threshold=2)) as runtime:
            expected = {(receipt.source_ref.source_type, receipt.source_ref.source_id) for receipt in receipts}
            await _wait_refs(runtime, sid, expected)
            await _stay_refs(runtime, sid, expected)
