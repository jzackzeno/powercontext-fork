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

"""HTTP scoring-service reranking for Memory search candidates."""

from __future__ import annotations

import math

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision, _validate_bounds
from powercontext.builtin.inference import InferenceUsage

_DEFAULT_POLICY_ID = "http-cross-encoder-v1"


class HTTPMemoryReranker:
    """Score Memory candidates through an HTTP reranking service.

    Candidates are sent in batches of at most ``batch_size`` using the
    caller's :class:`httpx.AsyncClient`. Global one-based ranks are preserved
    across batches, every rank is sent exactly once, and the final selection
    is a single global ordering by service score descending with ties broken
    by ascending original rank. ``InferenceUsage.requests`` reports the
    number of HTTP requests actually made.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = _DEFAULT_POLICY_ID,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Score every candidate and select the top ``limit`` global ranks."""

        _validate_bounds(len(candidates), limit)
        scores: dict[int, float] = {}
        requests = 0
        for start in range(0, len(candidates), self._batch_size):
            batch = candidates[start : start + self._batch_size]
            ranks = tuple(range(start + 1, start + len(batch) + 1))
            response = await self._client.post(
                self._endpoint,
                json={
                    "query": query,
                    "candidates": [
                        {"rank": rank, "text": candidate.text} for rank, candidate in zip(ranks, batch, strict=True)
                    ],
                },
            )
            response.raise_for_status()
            scores.update(_parse_batch_scores(response, expected_ranks=frozenset(ranks)))
            requests += 1
        ordered = sorted(scores, key=lambda rank: (-scores[rank], rank))
        return MemoryRerankDecision(
            selected_ranks=tuple(ordered[:limit]),
            usage=InferenceUsage(requests=requests),
        )


def _parse_batch_scores(response: httpx.Response, *, expected_ranks: frozenset[int]) -> dict[int, float]:
    """Validate one batch response and return its rank-to-score mapping.

    The body must contain exactly ``scores``; each score record must contain
    exactly ``rank`` and ``score``; ranks must cover this batch exactly once;
    and every score must be a finite number. Anything else is a ``ValueError``.
    """

    try:
        body = response.json()
    except ValueError as exc:
        raise ValueError("reranker response is not valid JSON") from exc  # noqa: TRY003
    if not isinstance(body, dict) or set(body) != {"scores"}:
        raise ValueError("reranker response must contain exactly 'scores'")  # noqa: TRY003
    raw_scores = body["scores"]
    if not isinstance(raw_scores, list):
        raise ValueError("reranker response 'scores' must be a list")  # noqa: TRY003, TRY004
    parsed: dict[int, float] = {}
    for item in raw_scores:
        if not isinstance(item, dict) or set(item) != {"rank", "score"}:
            raise ValueError("reranker score records must contain exactly 'rank' and 'score'")  # noqa: TRY003
        rank, score = item["rank"], item["score"]
        if type(rank) is not int or rank not in expected_ranks or rank in parsed:
            raise ValueError("reranker scores must cover this batch's ranks exactly once")  # noqa: TRY003
        if type(score) is bool or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("reranker scores must be finite numbers")  # noqa: TRY003
        parsed[rank] = float(score)
    if set(parsed) != set(expected_ranks):
        raise ValueError("reranker scores must cover this batch's ranks exactly once")  # noqa: TRY003
    return parsed


__all__ = ["HTTPMemoryReranker"]
