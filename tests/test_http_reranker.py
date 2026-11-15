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

"""Focused tests for the HTTP Memory reranker."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory.http_reranker import HTTPMemoryReranker
from powercontext.builtin.artifacts.memory.models import MemoryHit


def _hit(rank: int) -> MemoryHit:
    return MemoryHit(
        memory_ref=ArtifactRef(family="memory", artifact_id="m", revision=1),
        entry_id=f"e{rank}",
        entry_version_id=f"v{rank}",
        text=f"candidate {rank}",
        score=1.0 / rank,
        matched_by=("fts",),
    )


def test_batches_seven_and_globally_orders_cross_batch_ties() -> None:
    scores = {1: 0.1, 2: 0.9, 3: 0.2, 4: 0.3, 5: 0.9, 6: 0.4, 7: 0.5, 8: 0.99}
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert 1 <= len(body["candidates"]) <= 7
        seen.append(body)
        return httpx.Response(
            200,
            json={"scores": [{"rank": item["rank"], "score": scores[item["rank"]]} for item in body["candidates"]]},
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = HTTPMemoryReranker("https://reranker.invalid/score", client=client, batch_size=7)
            assert reranker.policy_id == "http-cross-encoder-v1"
            decision = await reranker.rerank("query", tuple(_hit(i) for i in range(1, 9)), 3)
        assert decision.selected_ranks == (8, 2, 5)
        assert decision.usage.requests == 2
        assert len(seen) == 2
        assert sorted(item["rank"] for request in seen for item in request["candidates"]) == list(range(1, 9))

    asyncio.run(scenario())


def test_rejects_malformed_json_body() -> None:
    async def scenario() -> None:
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, content=b"{not json"))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker("https://invalid", client=client).rerank("q", (_hit(1),), 1)

    asyncio.run(scenario())


def test_preserves_http_status_error() -> None:
    async def scenario() -> None:
        transport = httpx.MockTransport(lambda _request: httpx.Response(500, json={"error": "boom"}))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await HTTPMemoryReranker("https://invalid", client=client).rerank("q", (_hit(1),), 1)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "payload",
    [
        {"scores": [{"rank": 1, "score": "0.5"}]},
        {"scores": [{"rank": True, "score": 0.5}]},
        {"scores": {"rank": 1, "score": 0.5}},
        {"scores": [{"rank": 1, "score": 0.5}, {"rank": 1, "score": 0.5}]},
    ],
)
def test_rejects_non_numeric_or_duplicate_scores(payload) -> None:
    async def scenario() -> None:
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, content=json.dumps(payload).encode()))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker("https://invalid", client=client).rerank("q", (_hit(1),), 1)

    asyncio.run(scenario())


def test_accepts_custom_policy_id_and_full_limit() -> None:
    async def scenario() -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "scores": [
                        {"rank": item["rank"], "score": 0.5} for item in json.loads(request.content)["candidates"]
                    ]
                },
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            reranker = HTTPMemoryReranker("https://invalid", client=client, policy_id="custom-v2")
            assert reranker.policy_id == "custom-v2"
            decision = await reranker.rerank("q", (_hit(1), _hit(2)), 2)
        assert decision.selected_ranks == (1, 2)
        assert decision.usage.requests == 1

    asyncio.run(scenario())
