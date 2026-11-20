"""Plain-text, Markdown, source-code, XML and RTF parsing."""
from __future__ import annotations

import re
from pathlib import Path

from ..utils.encoding import read_text


def _strip_rtf(text: str) -> str:
    # Conservative RTF-to-text fallback; does not pretend to be a full RTF renderer.
    text = re.sub(r"\\'[0-9a-fA-F]{2}", "", text)
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", text)
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse(file_path: str | Path) -> dict:
    path = Path(file_path)
    text, encoding = read_text(path)
    if path.suffix.lower() == ".rtf":
        text = _strip_rtf(text)

    return {
        "text": text,
        "components": [{"type": "text", "text": text}],
        "metadata": {
            "parser": "direct_text",
            "encoding": encoding,
            "format": path.suffix.lower().lstrip("."),
        },
    }
