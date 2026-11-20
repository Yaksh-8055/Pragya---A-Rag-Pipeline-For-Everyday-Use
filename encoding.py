"""Bounded text encoding detection utilities."""
from __future__ import annotations

from pathlib import Path

try:
    import chardet
except ImportError:  # pragma: no cover - optional dependency
    chardet = None

DEFAULT_SAMPLE_SIZE = 64 * 1024


def detect_encoding(file_path: str | Path, sample_size: int = DEFAULT_SAMPLE_SIZE) -> dict:
    path = Path(file_path)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    with path.open("rb") as fh:
        sample = fh.read(sample_size)

    # BOMs are authoritative and avoid detector ambiguity.
    if sample.startswith(b"\xef\xbb\xbf"):
        return {"encoding": "utf-8-sig", "confidence": 1.0, "detector": "bom"}
    if sample.startswith(b"\xff\xfe\x00\x00"):
        return {"encoding": "utf-32", "confidence": 1.0, "detector": "bom"}
    if sample.startswith(b"\x00\x00\xfe\xff"):
        return {"encoding": "utf-32-be", "confidence": 1.0, "detector": "bom"}
    if sample.startswith(b"\xff\xfe"):
        return {"encoding": "utf-16", "confidence": 1.0, "detector": "bom"}
    if sample.startswith(b"\xfe\xff"):
        return {"encoding": "utf-16-be", "confidence": 1.0, "detector": "bom"}

    if chardet is not None and sample:
        result = chardet.detect(sample)
        encoding = result.get("encoding") or "utf-8"
        return {
            "encoding": encoding,
            "confidence": result.get("confidence"),
            "detector": "chardet",
        }

    return {"encoding": "utf-8", "confidence": None, "detector": "fallback"}


def read_text(file_path: str | Path, *, encoding: str | None = None) -> tuple[str, dict]:
    path = Path(file_path)
    info = detect_encoding(path) if encoding is None else {
        "encoding": encoding,
        "confidence": 1.0,
        "detector": "explicit",
    }

    enc = info["encoding"]
    try:
        return path.read_text(encoding=enc), info
    except (UnicodeDecodeError, LookupError):
        # A decoding fallback is preferable to silently losing a document.
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        info = {**info, "fallback": "utf-8-replace"}
        return text, info
