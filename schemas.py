"""Schemas and validation helpers for Stage 1 ingestion."""
from __future__ import annotations

from typing import Any


def validate_parser_result(value: Any) -> tuple[bool, str]:
    """Validate the stable parser contract: text/components/metadata."""
    if not isinstance(value, dict):
        return False, "Parser must return a dictionary."

    text = value.get("text", "")
    components = value.get("components", [])
    metadata = value.get("metadata", {})

    if not isinstance(text, str):
        return False, "Parser result 'text' must be a string."
    if not isinstance(components, list):
        return False, "Parser result 'components' must be a list."
    if not isinstance(metadata, dict):
        return False, "Parser result 'metadata' must be a dictionary."

    return True, ""


def normalize_parser_result(value: dict[str, Any]) -> dict[str, Any]:
    """Return a defensive copy of a valid parser result."""
    ok, error = validate_parser_result(value)
    if not ok:
        raise TypeError(error)

    return {
        "text": value.get("text", ""),
        "components": list(value.get("components", [])),
        "metadata": dict(value.get("metadata", {})),
    }
