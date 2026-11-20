"""OOXML spreadsheet and presentation parsing."""
from __future__ import annotations

from pathlib import Path


def _xlsx(path: Path) -> dict:
    from openpyxl import load_workbook

    workbook = load_workbook(str(path), read_only=True, data_only=True)
    components = []
    blocks = []
    try:
        for sheet in workbook.worksheets:
            rows = []
            for row in sheet.iter_rows(values_only=True):
                values = ["" if value is None else str(value) for value in row]
                if any(values):
                    rows.append(values)
                    blocks.append(f"[Sheet: {sheet.title}] " + " | ".join(values))
            components.append({"type": "worksheet", "name": sheet.title, "rows": rows})
    finally:
        workbook.close()

    return {
        "text": "\n".join(blocks),
        "components": components,
        "metadata": {"parser": "office", "format": "xlsx", "sheet_count": len(components)},
    }


def _pptx(path: Path) -> dict:
    from pptx import Presentation

    presentation = Presentation(str(path))
    components = []
    blocks = []
    for number, slide in enumerate(presentation.slides, start=1):
        slide_text = []
        for shape in slide.shapes:
            if not hasattr(shape, "text"):
                continue
            value = shape.text.strip()
            if value:
                slide_text.append(value)
        if slide_text:
            text = "\n".join(slide_text)
            blocks.append(f"[Slide {number}]\n{text}")
            components.append({"type": "slide", "slide": number, "text": text})

    return {
        "text": "\n\n".join(blocks),
        "components": components,
        "metadata": {"parser": "office", "format": "pptx", "slide_count": len(presentation.slides)},
    }


def parse(file_path: str | Path) -> dict:
    path = Path(file_path)
    if path.suffix.lower() == ".xlsx":
        return _xlsx(path)
    if path.suffix.lower() == ".pptx":
        return _pptx(path)
    raise ValueError(f"office parser does not support {path.suffix!r}")
