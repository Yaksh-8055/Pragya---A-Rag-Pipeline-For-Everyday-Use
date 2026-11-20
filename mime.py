"""MIME and lightweight magic-signature detection."""
from __future__ import annotations

import mimetypes
from pathlib import Path

MAGIC_MIME = {
    b"%PDF-": "application/pdf",
    b"PK\x03\x04": "application/zip",  # DOCX/XLSX/PPTX are OOXML ZIPs.
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    b"BM": "image/bmp",
    b"II*\x00": "image/tiff",
    b"MM\x00*": "image/tiff",
}


def extension_mime(path: str | Path) -> str:
    mime, _ = mimetypes.guess_type(Path(path).name)
    return mime or "application/octet-stream"


def sniff_mime(path: str | Path) -> str:
    p = Path(path)
    with p.open("rb") as fh:
        head = fh.read(32)

    for signature, mime in MAGIC_MIME.items():
        if head.startswith(signature):
            return mime

    return extension_mime(p)
