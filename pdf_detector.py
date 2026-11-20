"""PDF text-layer detection without mutating global state."""
from __future__ import annotations

from pathlib import Path


def has_text_layer(file_path: str | Path, min_non_whitespace_chars: int = 1) -> bool:
    if min_non_whitespace_chars < 1:
        raise ValueError("min_non_whitespace_chars must be >= 1")

    import fitz  # PyMuPDF

    document = fitz.open(str(file_path))
    try:
        total = 0
        for page in document:
            total += len(page.get_text("text").strip())
            if total >= min_non_whitespace_chars:
                return True
        return False
    finally:
        document.close()
