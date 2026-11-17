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

"""Focused behavior tests for query-driven evidence selection."""

from __future__ import annotations

import json
import re

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.artifacts.memory import MemoryHit
from powercontext.builtin.runtime.evidence_selection import score_sentences, split_sentences, tokenize
from powercontext.builtin.runtime.models import PrepareContextRequest
from powercontext.builtin.runtime.prepared_context import (
    MemoryEntryAddress,
    PreparedContextBuilder,
    PreparedMemoryCandidates,
)

_MEMORY_REF = ArtifactRef(family="memory", artifact_id="art_memory", revision=1)


def _hit(entry_id: str, text: str) -> MemoryHit:
    return MemoryHit(
        memory_ref=_MEMORY_REF,
        entry_id=entry_id,
        entry_version_id=entry_id + "-v1",
        text=text,
        score=1.0,
        matched_by=("fts",),
    )


def _build(query: str, documents: tuple[str, ...], max_bytes: int = 1800, *, scope_id: str = "scp_test") -> object:
    groups = (
        PreparedMemoryCandidates(
            scope_id=scope_id,
            memory_ref=_MEMORY_REF,
            hits=tuple(_hit(f"doc-{index}", text) for index, text in enumerate(documents)),
        ),
    )
    request = PrepareContextRequest(query=query, max_bytes=max_bytes, evidence_selection=True)
    return PreparedContextBuilder().build_scopes_result(
        request=request, current_scope_id=scope_id, memory_candidates=groups
    )


def _token_recall(selected: str, gold: str) -> float:
    pattern = re.compile(r"[a-z]+|\d+")
    selected_tokens = sorted(pattern.findall(selected.lower()))
    gold_tokens = sorted(pattern.findall(gold.lower()))
    remaining = list(selected_tokens)
    hits = 0
    for token in gold_tokens:
        if token in remaining:
            hits += 1
            remaining.remove(token)
    return hits / len(gold_tokens) if gold_tokens else 1.0


def test_split_sentences_keeps_abbreviations_and_decimals() -> None:
    text = "The dose is 3.5 mg, e.g. twice daily. Dr. Smith reviewed Fig. 3."
    assert [text[start:end] for start, end in split_sentences(text)] == [
        "The dose is 3.5 mg, e.g. twice daily.",
        "Dr. Smith reviewed Fig. 3.",
    ]


def test_split_sentences_handles_cjk_and_bare_text() -> None:
    text = "海洋观测 station records temperatures. 海洋观测 station records humidity."
    spans = split_sentences(text)
    assert len(spans) == 2
    assert [text[start:end] for start, end in spans] == [
        "海洋观测 station records temperatures.",
        "海洋观测 station records humidity.",
    ]
    assert split_sentences("a title without punctuation") == ((0, 27),)


def test_tokenize_and_bm25_rank_distinctive_terms_first() -> None:
    assert tokenize("Cobalt Sensor 37") == {"cobalt": 1, "sensor": 1, "37": 1}
    sentences = (
        "General administrative notes about the facility.",
        "The cobalt sensor is calibrated at 37 degrees.",
        "The amber valve opens at 19 degrees.",
    )
    scores = score_sentences("cobalt sensor calibrated degrees", sentences)
    assert scores[1] > scores[2] > scores[0] == 0.0


def test_query_change_flips_selection_to_late_evidence() -> None:
    prefix = "General administrative notes. " * 70
    documents = (
        prefix + "The cobalt sensor is calibrated at 37 degrees. ",
        prefix + "The amber valve opens at 19 degrees. ",
    )
    first = _build("cobalt sensor calibrated", documents, max_bytes=1200)
    assert any("37" in fragment.text for fragment in first.fragments)
    assert all("19" not in fragment.text for fragment in first.fragments)
    second = _build("amber valve opens", documents, max_bytes=1200)
    assert any("19" in fragment.text for fragment in second.fragments)
    assert all("37" not in fragment.text for fragment in second.fragments)


def test_fragments_carry_exact_spans_citations_and_origins() -> None:
    result = _build("cobalt sensor calibrated", ("The cobalt sensor is calibrated at 37 degrees. ",), max_bytes=1200)
    assert len(result.fragments) == 1
    fragment = result.fragments[0]
    assert isinstance(fragment.address, MemoryEntryAddress)
    assert fragment.address.memory.scope_id == "scp_test"
    assert fragment.text == "The cobalt sensor is calibrated at 37 degrees."
    assert fragment.address.entry_id == "doc-0"
    items = json.loads(result.context.content.splitlines()[-2])["items"]
    assert len(items) == len(result.fragments) == len(result.origins)
    assert items[0]["content"] == fragment.text
    assert items[0]["start"] == fragment.start and items[0]["end"] == fragment.end
    assert items[0]["citation"] == {
        "scope_id": "scp_test",
        "memory_ref": _MEMORY_REF.model_dump(mode="json"),
        "entry_id": "doc-0",
        "entry_version_id": "doc-0-v1",
    }
    assert result.origins[0] == MemoryCitation(memory_ref=_MEMORY_REF, entry_id="doc-0", entry_version_id="doc-0-v1")
    assert result.context.content_bytes == len(result.context.content.encode("utf-8")) <= 1200
    assert "BEGIN_POWERCONTEXT_PREPARED_CONTEXT_V1" in result.context.content
    assert "END_POWERCONTEXT_PREPARED_CONTEXT_V1" in result.context.content


def test_adjacent_sentences_merge_into_one_fragment() -> None:
    text = (
        "The cobalt sensor is calibrated at 37 degrees. "
        "Calibration uses a cobalt reference standard. "
        "Unrelated administrative notes follow here."
    )
    result = _build("cobalt sensor calibrated reference standard", (text,), max_bytes=1800)
    assert len(result.fragments) == 1
    assert "37 degrees. Calibration uses a cobalt reference standard." in result.fragments[0].text


def test_unicode_content_is_charged_as_utf8_and_budget_holds() -> None:
    result = _build("海洋观测 station", ("海洋观测 station records temperatures. " * 40,), max_bytes=1200)
    assert result.fragments
    assert result.context.content_bytes == len(result.context.content.encode("utf-8")) <= 1200


def test_no_matching_sentence_returns_empty_context() -> None:
    result = _build("quantum entanglement", ("The cobalt sensor is calibrated at 37 degrees. ",), max_bytes=1200)
    assert result.context.status == "empty"
    assert result.context.content is None
    assert result.fragments == ()
    assert result.origins == ()


def test_default_prepare_is_unchanged_and_has_no_fragments() -> None:
    groups = (
        PreparedMemoryCandidates(
            scope_id="scp_test",
            memory_ref=_MEMORY_REF,
            hits=(_hit("doc-0", "A station records temperature."),),
        ),
    )
    request = PrepareContextRequest(query="temperature", max_bytes=1200)
    result = PreparedContextBuilder().build_scopes_result(
        request=request, current_scope_id="scp_test", memory_candidates=groups
    )
    assert result.fragments == ()
    items = json.loads(result.context.content.splitlines()[-2])["items"]
    assert items[0]["content"] == "A station records temperature."


def test_cross_scope_fragment_uses_address_origin() -> None:
    other_ref = ArtifactRef(family="memory", artifact_id="art_other", revision=1)
    groups = (
        PreparedMemoryCandidates(
            scope_id="scp_test",
            memory_ref=_MEMORY_REF,
            hits=(_hit("doc-0", "The cobalt sensor is calibrated at 37 degrees. "),),
        ),
        PreparedMemoryCandidates(
            scope_id="scp_other",
            memory_ref=other_ref,
            hits=(
                MemoryHit(
                    memory_ref=other_ref,
                    entry_id="doc-9",
                    entry_version_id="doc-9-v1",
                    text="The amber valve opens at 19 degrees. ",
                    score=1.0,
                    matched_by=("fts",),
                ),
            ),
        ),
    )
    request = PrepareContextRequest(query="amber valve opens", max_bytes=1200, evidence_selection=True)
    result = PreparedContextBuilder().build_scopes_result(
        request=request, current_scope_id="scp_test", memory_candidates=groups
    )
    assert len(result.fragments) == 1
    assert result.fragments[0].address.memory.scope_id == "scp_other"
    assert result.origins[0] == result.fragments[0].address


def test_synthetic_three_document_recall_and_full_coverage() -> None:
    """A planted-gold stand-in for the hidden quality gate.

    Three abstracts, two gold sentences per request (one late, one
    adjacent pair), budgets 1400 and 1800. Measures the same token recall
    and full-span coverage the hidden gate uses.
    """

    filler = "General background on laboratory procedures and safety notes. " * 6
    documents = (
        filler + "Quantum dots show inductive properties in 0-dimensional biomaterials. "
        "The dots were synthesized with a cobalt coating for tracking. ",
        filler + "Stem cell differentiation improved with nanoscale scaffolds. "
        "Scaffolds made from 0-dimensional biomaterials induced differentiation. ",
        filler + "Magnetic nanoparticles enable in vivo tracking of stem cells. "
        "No inductive effect was observed in the control group. ",
    )
    gold = (
        "Quantum dots show inductive properties in 0-dimensional biomaterials. "
        "The dots were synthesized with a cobalt coating for tracking. "
        "Scaffolds made from 0-dimensional biomaterials induced differentiation."
    )
    for max_bytes in (1400, 1800):
        result = _build("0-dimensional biomaterials show inductive properties", documents, max_bytes=max_bytes)
        selected = " ".join(fragment.text for fragment in result.fragments)
        assert _token_recall(selected, gold) >= 0.65
        assert result.context.content_bytes <= max_bytes
        # Every gold sentence is fully covered by the selected spans.
        for sentence in (
            "Quantum dots show inductive properties in 0-dimensional biomaterials.",
            "The dots were synthesized with a cobalt coating for tracking.",
            "Scaffolds made from 0-dimensional biomaterials induced differentiation.",
        ):
            assert sentence in selected
