"""CSV/JSON parsing while preserving every source record for RAG."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from ..utils.encoding import read_text


def _csv_to_text(text: str) -> tuple[str, list[list[str]]]:
    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.excel

    reader = csv.reader(io.StringIO(text), dialect=dialect)
    rows: list[list[str]] = []
    lines: list[str] = []

    # Deliberately use csv.reader rather than DictReader. A CSV may be
    # headerless, and the first record must never be silently discarded.
    for row in reader:
        values = ["" if value is None else str(value) for value in row]
        if not any(value.strip() for value in values):
            continue
        rows.append(values)
        lines.append(" | ".join(values))

    return "\n".join(lines), rows


def parse(file_path: str | Path) -> dict:
    path = Path(file_path)
    text, encoding = read_text(path)
    ext = path.suffix.lower()

    if ext == ".json":
        data = json.loads(text)
        rendered = json.dumps(data, ensure_ascii=False, indent=2)
        component_type = "json"
        components = [{"type": component_type, "text": rendered}]
        extra = {"json_type": type(data).__name__}
    else:
        rendered, rows = _csv_to_text(text)
        component_type = "csv"
        components = [{"type": component_type, "text": rendered}]
        extra = {"row_count": len(rows), "rows": rows}

    return {
        "text": rendered,
        "components": components,
        "metadata": {
            "parser": "structured",
            "encoding": encoding,
            "format": ext.lstrip("."),
            **extra,
        },
    }

