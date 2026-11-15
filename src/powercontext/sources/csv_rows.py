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

"""CSV row-level Sources: one captured Source per CSV record.

Each data row of a CSV document becomes an independent ``CAPTURED``
:class:`CSVRowSource` whose identity is derived from the document id and
the row's primary-key value. The row keeps the original header order and
the original string values exactly as parsed: no trimming and no type
conversion.
"""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    AdapterSourceDefinition,
    SourceDefinition,
    TextEvidence,
)
from powercontext.sources.models import Source, SourceMaterialization

CSV_ROW_SOURCE_NAME = "csv-row"
_CSV_ROW_SOURCE_ID_PREFIX = "csv:"


class CSVDocumentInput(BaseModel):
    """A whole offline CSV document to parse into rows."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One parsed CSV row, preserving header order and raw string values."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """A captured CSV row that can be used as Artifact evidence."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Build the stable identity for one CSV row.

    Both components are percent-encoded separately with
    ``urllib.parse.quote(component, safe="")`` so neither can collide with
    the ``:`` separators. The identity must not exceed 256 characters.
    """

    if not isinstance(document_id, str) or not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be a non-empty string without leading or trailing whitespace")  # noqa: TRY003
    if not isinstance(primary_key_value, str) or not primary_key_value.strip():
        raise ValueError("primary_key_value must be a non-empty string that is not only whitespace")  # noqa: TRY003
    identity = f"{_CSV_ROW_SOURCE_ID_PREFIX}{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"csv row source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return identity


def _decode_csv_text(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV content must be valid UTF-8") from error  # noqa: TRY003


def _read_csv_records(text: str) -> list[list[str]]:
    try:
        return list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as error:
        raise ValueError(f"malformed CSV content: {error}") from error  # noqa: TRY003


def _validate_header(header: list[str], primary_key: str) -> int:
    if not header:
        raise ValueError("CSV header must not be empty")  # noqa: TRY003
    seen_columns: set[str] = set()
    for column in header:
        if not column.strip():
            raise ValueError("CSV header fields must be non-empty and not only whitespace")  # noqa: TRY003
        if column in seen_columns:
            raise ValueError(f"CSV header field {column!r} is duplicated")  # noqa: TRY003
        seen_columns.add(column)
    if primary_key not in seen_columns:
        raise ValueError(f"primary_key {primary_key!r} is not a header field")  # noqa: TRY003
    return header.index(primary_key)


def _parse_data_rows(
    records: list[list[str]],
    header: list[str],
    primary_key_index: int,
    document_id: str,
    primary_key: str,
) -> tuple[CSVRowInput, ...]:
    rows: list[CSVRowInput] = []
    seen_primary_key_values: set[str] = set()
    for record in records:
        if record == []:
            continue
        if len(record) != len(header):
            raise ValueError(  # noqa: TRY003
                f"CSV record has {len(record)} fields, expected {len(header)}"
            )
        primary_key_value = record[primary_key_index]
        if not primary_key_value.strip():
            raise ValueError("CSV primary key value must not be empty or only whitespace")  # noqa: TRY003
        if primary_key_value in seen_primary_key_values:
            raise ValueError(f"CSV primary key value {primary_key_value!r} is duplicated")  # noqa: TRY003
        seen_primary_key_values.add(primary_key_value)
        # Also enforces the 256-character identity limit for this row.
        csv_row_source_id(document_id, primary_key_value)
        rows.append(
            CSVRowInput(
                document_id=document_id,
                primary_key=primary_key,
                primary_key_value=primary_key_value,
                columns=tuple(header),
                values=tuple(record),
            )
        )
    return tuple(rows)


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse a whole CSV document into one :class:`CSVRowInput` per row.

    The document is decoded as strict UTF-8 (a leading UTF-8 BOM is
    accepted) and parsed with the standard library ``csv.reader`` using
    ``strict=True``. The first record is the header; completely empty
    physical records after the header are ignored. The whole document is
    parsed and validated before anything is returned, so any error fails
    atomically with :class:`ValueError` and no partial rows are produced.
    """

    document_id = value.document_id
    primary_key = value.primary_key
    if not isinstance(document_id, str) or not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be a non-empty string without leading or trailing whitespace")  # noqa: TRY003
    if not isinstance(primary_key, str) or not primary_key or not primary_key.strip():
        raise ValueError("primary_key must be a non-empty string")  # noqa: TRY003
    records = _read_csv_records(_decode_csv_text(value.content))
    if not records:
        raise ValueError("CSV document must not be empty")  # noqa: TRY003
    primary_key_index = _validate_header(records[0], primary_key)
    return _parse_data_rows(records[1:], records[0], primary_key_index, document_id, primary_key)


def _validate_row_input(value: CSVRowInput) -> str:
    """Validate a row for materialization and return its identity."""

    if (
        not isinstance(value.document_id, str)
        or not value.document_id
        or value.document_id != value.document_id.strip()
    ):
        raise ValueError("document_id must be a non-empty string without leading or trailing whitespace")  # noqa: TRY003
    if not isinstance(value.primary_key, str) or not value.primary_key or not value.primary_key.strip():
        raise ValueError("primary_key must be a non-empty string")  # noqa: TRY003
    if not value.columns:
        raise ValueError("CSV row columns must not be empty")  # noqa: TRY003
    seen_columns: set[str] = set()
    for column in value.columns:
        if not column.strip():
            raise ValueError("CSV header fields must be non-empty and not only whitespace")  # noqa: TRY003
        if column in seen_columns:
            raise ValueError(f"CSV header field {column!r} is duplicated")  # noqa: TRY003
        seen_columns.add(column)
    if value.primary_key not in seen_columns:
        raise ValueError(f"primary_key {value.primary_key!r} is not a header field")  # noqa: TRY003
    if len(value.values) != len(value.columns):
        raise ValueError(  # noqa: TRY003
            f"CSV row has {len(value.values)} values, expected {len(value.columns)}"
        )
    if value.values[value.columns.index(value.primary_key)] != value.primary_key_value:
        raise ValueError("CSV primary key value does not match the value in the primary key column")  # noqa: TRY003
    return csv_row_source_id(value.document_id, value.primary_key_value)


class CSVRowSourceAdapter:
    """Materialize one parsed CSV row as a captured Source."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        identity = _validate_row_input(value)
        return CSVRowSource(
            name=identity,
            source_id=identity,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    """Expose one captured CSV row as standard text evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.name,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_ADAPTER = CSVRowSourceAdapter()
CSV_ROW_SOURCE_DEFINITION: SourceDefinition[CSVRowInput, CSVRowSource, CSVRowInput] = AdapterSourceDefinition(
    CSV_ROW_SOURCE_ADAPTER,
    projections=(CSVRowTextEvidenceProjection(),),
)

__all__ = [
    "CSV_ROW_SOURCE_ADAPTER",
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_NAME",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "CSVRowSourceAdapter",
    "CSVRowTextEvidenceProjection",
    "csv_row_source_id",
    "parse_csv_document",
]
