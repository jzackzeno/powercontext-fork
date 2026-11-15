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

"""Focused behavior tests for CSV row-level Sources."""

from __future__ import annotations

import asyncio

import pytest

from powercontext.sources import SourceDefinitionRegistry, TextEvidence
from powercontext.sources.csv_rows import (
    CSV_ROW_SOURCE_DEFINITION,
    CSVDocumentInput,
    CSVRowSource,
    csv_row_source_id,
    parse_csv_document,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


def _parse(content: bytes, document_id: str = "airports", primary_key: str = "id"):
    return parse_csv_document(CSVDocumentInput(document_id=document_id, primary_key=primary_key, content=content))


def test_parse_preserves_raw_values_and_header_order() -> None:
    rows = _parse(b'id,name,note\n1, Airport,"a, b"\n')
    assert rows[0].columns == ("id", "name", "note")
    assert rows[0].values == ("1", " Airport", "a, b")


def test_parse_accepts_bom_and_ignores_empty_records() -> None:
    rows = _parse(b"\xef\xbb\xbfid,name\n1,one\n\n2,two\n")
    assert [row.primary_key_value for row in rows] == ["1", "2"]
    assert _parse(b"id,name\n") == ()


def test_parse_rejects_invalid_documents_atomically() -> None:
    for content in (
        b"",
        b"id,name\n1,one\n1,duplicate\n",
        b"id,name\n,empty-key\n",
        b"id,name\n1\n",
        b"id,id\n1,1\n",
        b'id,name\n1,"unclosed\n',
        b"id,name\n\xff,invalid utf-8\n",
    ):
        with pytest.raises(ValueError):
            _parse(content)


def test_row_identity_is_delimiter_safe_and_bounded() -> None:
    assert csv_row_source_id("doc:one", "key/one") == "csv:doc%3Aone:key%2Fone"
    assert len(csv_row_source_id("d" * 250, "1")) == 256
    with pytest.raises(ValueError):
        csv_row_source_id("d" * 251, "1")


def test_resolve_and_project_one_row() -> None:
    row = _parse(b"id,name\n1,Total RF Heliport\n")[0]
    registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
    assert CSV_ROW_SOURCE_DEFINITION.name == "csv-row"
    assert CSV_ROW_SOURCE_DEFINITION.version == "1"
    source = asyncio.run(registry.resolve(row))
    assert isinstance(source, CSVRowSource)
    assert source.source_id == source.name == "csv:airports:1"
    assert asyncio.run(registry.read(source)) == row
    evidence = TextEvidence.model_validate(registry.project(source, TEXT_EVIDENCE_PROJECTION_KEY))
    assert evidence.source_type == "csv-row"
    assert evidence.source_id == "csv:airports:1"
    assert evidence.content == '{"id":"1","name":"Total RF Heliport"}'
    assert evidence.metadata == {
        "document_id": "airports",
        "primary_key": "id",
        "primary_key_value": "1",
        "columns": ["id", "name"],
        "row": {"id": "1", "name": "Total RF Heliport"},
    }
