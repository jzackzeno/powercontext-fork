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

"""Query-driven sentence selection over Memory entry text.

The selector is pure text work: split an entry into sentences with exact
Python character offsets, rank them against the query with BM25, and let
the caller pack the winners into the byte budget. Fragments are always
verbatim spans of the original text, never rewritten or summarized.
"""

from __future__ import annotations

import math
import re
from collections import Counter

__all__ = ["score_sentences", "split_sentences", "tokenize"]

_TOKEN_PATTERN = re.compile(r"[a-z]+|\d+|[\u4e00-\u9fff]")
_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*")
# Periods that never end a sentence: decimals and common abbreviations.
_ABBREVIATIONS = frozenset({
    "al",
    "approx",
    "dr",
    "e.g",
    "eq",
    "etc",
    "fig",
    "i.e",
    "jr",
    "mr",
    "mrs",
    "ms",
    "no",
    "prof",
    "sr",
    "st",
    "u.k",
    "u.s",
    "vs",
})
# Closed-class words carry no evidence signal; dropping them keeps generic
# sentences from outscoring the sentences that share the query's content
# terms. The list is deliberately conservative: verbs and nouns stay.
_STOPWORDS = frozenset({
    "a",
    "about",
    "after",
    "again",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "during",
    "each",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "me",
    "more",
    "most",
    "my",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "on",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
})


def tokenize(text: str) -> Counter[str]:
    """Split text into lowercase word, number, and CJK-character tokens."""

    return Counter(_TOKEN_PATTERN.findall(text.lower()))


def split_sentences(text: str) -> tuple[tuple[int, int], ...]:
    """Split text into sentences as half-open Python character spans.

    Spans are trimmed of surrounding whitespace and never overlap; text
    without sentence punctuation is one sentence. A period does not end a
    sentence inside a decimal or after a known abbreviation or initial.
    """

    spans: list[tuple[int, int]] = []
    start = 0
    for line_start, line_end in _lines(text):
        start = line_start
        for match in _SENTENCE_END.finditer(text, line_start, line_end):
            if _is_sentence_end(text, match):
                end = match.end()
                spans.append(_trim_span(text, start, end))
                start = end
        if start < line_end:
            spans.append(_trim_span(text, start, line_end))
            start = line_end
    return tuple(span for span in spans if span[0] < span[1])


def score_sentences(query: str, sentences: tuple[str, ...]) -> tuple[float, ...]:
    """Rank sentences against the query with BM25 over the sentence set.

    The query is a claim, so every distinct content term counts once;
    sentence term frequency saturates and longer sentences are penalized,
    which keeps the byte budget spent on evidence-dense sentences.
    """

    query_terms = {term for term in tokenize(query) if term not in _STOPWORDS}
    if not query_terms or not sentences:
        return tuple(0.0 for _ in sentences)
    counts = [tokenize(sentence) for sentence in sentences]
    lengths = [sum(counter.values()) for counter in counts]
    average_length = sum(lengths) / len(lengths) if lengths else 1.0
    if average_length <= 0:
        average_length = 1.0
    document_count = len(sentences)
    scores: list[float] = []
    for counter, length in zip(counts, lengths, strict=True):
        score = 0.0
        for term in query_terms:
            frequency = counter.get(term, 0)
            if not frequency:
                continue
            documents_with_term = sum(1 for other in counts if term in other)
            idf = math.log(1.0 + (document_count - documents_with_term + 0.5) / (documents_with_term + 0.5))
            score += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / average_length))
        scores.append(score)
    return tuple(scores)


def _lines(text: str) -> tuple[tuple[int, int], ...]:
    lines: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"\n", text):
        lines.append((start, match.start()))
        start = match.end()
    lines.append((start, len(text)))
    return tuple(lines)


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _is_sentence_end(text: str, match: re.Match[str]) -> bool:
    end = match.end()
    # A decimal point never ends a sentence.
    if "." in match.group() and end > 1 and text[end - 2].isdigit() and end < len(text) and text[end].isdigit():
        return False
    # The punctuation must be followed by whitespace or the end of the
    # text. The next sentence may start with any character: requiring an
    # uppercase letter would miss boundaries before CJK or lowercase text.
    if end < len(text) and not text[end].isspace():
        return False
    token_start = match.start()
    while token_start > 0 and (text[token_start - 1].isalnum() or text[token_start - 1] in "."):
        token_start -= 1
    token = text[token_start : match.start()].lower()
    return token not in _ABBREVIATIONS and not (len(token) == 1 and token.isalpha())
