"""OCR parser for raster images and scanned PDFs."""
from __future__ import annotations

from pathlib import Path


def _ocr_image(path: Path) -> dict:
    from PIL import Image, ImageSequence
    import pytesseract

    components = []
    blocks = []
    with Image.open(path) as image:
        frame_count = getattr(image, "n_frames", 1)
        for frame_number, frame in enumerate(ImageSequence.Iterator(image), start=1):
            frame = frame.convert("RGB")
            text = pytesseract.image_to_string(frame).strip()
            if text:
                blocks.append(text)
                components.append({
                    "type": "ocr_frame" if frame_count > 1 else "ocr",
                    "frame": frame_number,
                    "text": text,
                    "source": path.name,
                })

    return {
        "text": "\n\n".join(blocks),
        "components": components,
        "metadata": {
            "parser": "ocr",
            "format": path.suffix.lower().lstrip("."),
            "frame_count": frame_count,
        },
    }


def _ocr_pdf(path: Path) -> dict:
    import fitz
    import pytesseract
    from PIL import Image

    document = fitz.open(str(path))
    components = []
    blocks = []
    page_count = len(document)
    try:
        for number, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            text = pytesseract.image_to_string(image).strip()
            if text:
                blocks.append(text)
                components.append({"type": "ocr_page", "page": number, "text": text})
    finally:
        document.close()

    return {
        "text": "\n\n".join(blocks),
        "components": components,
        "metadata": {"parser": "ocr", "format": "pdf", "page_count": page_count},
    }


def parse(file_path: str | Path) -> dict:
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        return _ocr_pdf(path)
    return _ocr_image(path)
