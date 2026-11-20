"""Layout-aware parsing for DOCX, PDF and HTML  (LangChain-assisted).

DROP-IN replacement for  src/stage_1_ingestion/parsers/layout.py
Same module path, same parse(file_path) entry point, same return contract
{"text": str, "components": list, "metadata": dict}. Nothing else in the
pipeline needs to change.

What is different from the old parser
  DOCX  - reads the document IN ORDER (tables stay where they were),
          turns Word headings into Markdown (# / ## / ###) so Stage 2 can build
          section_headers, keeps merged table cells as "a | b" rows,
          then asks LangChain's Docx2txtLoader for a second opinion and
          appends anything the structured pass missed (headers/footers,
          text boxes, content controls, nested tables),
          and OCRs screenshots embedded in the file (if Tesseract is installed).
  PDF   - uses LangChain's PyMuPDFLoader, falling back to PyPDFLoader and then to
          plain PyMuPDF, so one broken library no longer fails the file.
  HTML  - unchanged (LangChain's HTML loader just wraps the same BeautifulSoup
          call and adds nothing).

Compatible by design
  * LangChain, docx2txt, Tesseract are all OPTIONAL. If any is missing or fails
    to import (e.g. on a very new Python), the parser silently falls back and
    still returns valid output. Only a file that is genuinely unreadable raises,
    so the router reports the real reason.
  * Python 3.9+, Windows paths, no new required dependency.
  * Optional installs:   pip install langchain-community docx2txt
"""
from __future__ import annotations

import io
import re
import warnings
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
USE_LANGCHAIN = True          # False = never touch LangChain
OCR_DOCX_IMAGES = True        # OCR screenshots inside .docx files
MAX_OCR_IMAGES = 25           # per document, to bound processing time
MIN_IMAGE_SIDE_PX = 80        # ignore icons / bullets
OCR_TIMEOUT_S = 30            # per image

_MD_HEADER = re.compile(r"^#{1,6}\s")
_LOADERS: dict = {}


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _langchain_loader(name: str):
    """Return a LangChain document-loader class, or None if unavailable."""
    if not USE_LANGCHAIN:
        return None
    if name not in _LOADERS:
        loader = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from langchain_community import document_loaders
                loader = getattr(document_loaders, name)
        except Exception:  # ImportError, pydantic problems on new Pythons, anything
            loader = None
        _LOADERS[name] = loader
    return _LOADERS[name]


def _escape_lines(text: str) -> str:
    """Stop body text (e.g. '# comment' in pasted Python) being mistaken for a
    Markdown heading by Stage 2. A leading space is enough."""
    return "\n".join((" " + ln) if _MD_HEADER.match(ln) else ln for ln in text.split("\n"))


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s).strip().lower()


def _missing_lines(other_text: str, base_text: str, ignore: set | None = None) -> list:
    """Lines present in other_text but not already contained in base_text."""
    blob = _norm(base_text)
    seen = set(ignore or ())
    out = []
    for line in other_text.splitlines():
        line = line.strip()
        n = _norm(line)
        if len(line) < 2 or not n or n in seen:
            continue
        seen.add(n)
        if n not in blob:
            out.append(line)
    return out


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------
def _heading_level(style_name: str) -> int:
    m = re.match(r"(?i)^heading\s*(\d+)$", style_name or "")
    if m:
        return max(1, min(int(m.group(1)), 6))
    return 1 if (style_name or "").strip().lower() == "title" else 0


def _docx_structure(path: Path):
    """Walk the body in document order. Returns (blocks, components, counts)."""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    blocks, components = [], []
    headings = tables = 0

    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]

        if tag == "p":
            para = Paragraph(child, doc)
            text = para.text.strip()
            if not text:
                continue
            style = getattr(para.style, "name", None) or ""
            level = _heading_level(style)
            if level:
                blocks.append("#" * level + " " + " ".join(text.split()))
                components.append({"type": "heading", "level": level, "text": text, "style": style})
                headings += 1
            else:
                body = _escape_lines(text)
                if style.lower().startswith("list bullet"):
                    body = "- " + body
                blocks.append(body)
                components.append({"type": "paragraph", "text": text, "style": style})

        elif tag == "tbl":
            table = Table(child, doc)
            rows = []
            for row in table.rows:
                cells, seen = [], []
                for cell in row.cells:
                    if any(cell._tc is s for s in seen):   # merged cell repeats itself
                        continue
                    seen.append(cell._tc)
                    cells.append(" ".join(cell.text.split()))
                if any(cells):
                    rows.append(cells)
            if rows:
                blocks.append(_escape_lines("\n".join(" | ".join(r) for r in rows)))
                components.append({"type": "table", "index": tables, "rows": rows})
                tables += 1

    counts = {"paragraph_count": len(doc.paragraphs), "table_count": len(doc.tables),
              "heading_count": headings}
    return blocks, components, counts


def _docx_langchain_text(path: Path) -> str:
    loader = _langchain_loader("Docx2txtLoader")
    if loader is None:
        raise RuntimeError("langchain_community Docx2txtLoader is not available")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        docs = loader(str(path)).load()
    return "\n".join(d.page_content or "" for d in docs)


def _ocr_docx_images(path: Path):
    """OCR screenshots/diagrams inside a .docx. Returns (results, notes)."""
    if not OCR_DOCX_IMAGES:
        return [], []
    try:
        from PIL import Image
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception:
        return [], ["image OCR skipped (Pillow/pytesseract/Tesseract not available)"]

    def natural(name: str):
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]

    results, notes = [], []
    try:
        with zipfile.ZipFile(path) as z:
            names = sorted((n for n in z.namelist()
                            if n.startswith("word/media/")
                            and n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"))),
                           key=natural)
            if len(names) > MAX_OCR_IMAGES:
                notes.append(f"only the first {MAX_OCR_IMAGES} of {len(names)} images were OCR'd")
            for name in names[:MAX_OCR_IMAGES]:
                try:
                    with Image.open(io.BytesIO(z.read(name))) as im:
                        if min(im.size) < MIN_IMAGE_SIDE_PX:
                            continue
                        im = im.convert("RGB")
                        if im.width < 800:                       # small screenshots OCR better enlarged
                            im = im.resize((im.width * 2, im.height * 2))
                        text = pytesseract.image_to_string(im, timeout=OCR_TIMEOUT_S).strip()
                except Exception:
                    continue
                if len(text) >= 8:
                    results.append((name.rsplit("/", 1)[-1], text))
    except (zipfile.BadZipFile, OSError):
        pass
    return results, notes


def _docx(path: Path) -> dict:
    notes = []

    # 1) structure (python-docx): order, headings, tables
    structure_error = None
    try:
        blocks, components, counts = _docx_structure(path)
    except Exception as exc:
        structure_error, blocks, components, counts = exc, [], [], {}

    # 2) coverage (LangChain / docx2txt): everything that is text in the file
    lc_text, lc_error = "", None
    try:
        lc_text = _docx_langchain_text(path)
    except Exception as exc:
        lc_error = exc

    if structure_error is not None and lc_error is not None:
        raise structure_error        # genuinely unreadable: let the router report the real reason

    if structure_error is not None:  # python-docx failed, LangChain could read it
        notes.append(f"structure pass failed ({type(structure_error).__name__}); used LangChain text only")
        cleaned = _escape_lines(re.sub(r"\n{3,}", "\n\n", lc_text).strip())
        blocks = [cleaned] if cleaned else []
        components = [{"type": "paragraph", "text": cleaned, "style": None}] if cleaned else []
        counts = {"paragraph_count": 0, "table_count": 0, "heading_count": 0}

    parts = list(blocks)
    base_text = "\n\n".join(parts)

    extra = _missing_lines(lc_text, base_text) if lc_text and structure_error is None else []
    if extra:
        parts.append("## Additional text (headers, text boxes, other content)\n\n"
                     + _escape_lines("\n".join(extra)))
        components.append({"type": "additional_text", "text": "\n".join(extra)})

    ocr_results, ocr_notes = _ocr_docx_images(path)
    notes.extend(ocr_notes)
    if ocr_results:
        parts.append("## Text found in images\n\n" + "\n\n".join(
            f"[Image: {name}]\n{_escape_lines(text)}" for name, text in ocr_results))
        components.extend({"type": "image_ocr", "image": n, "text": t} for n, t in ocr_results)

    if lc_error is not None:
        notes.append(f"LangChain unavailable or failed: {type(lc_error).__name__}: {lc_error}")

    return {
        "text": "\n\n".join(parts),
        "components": components,
        "metadata": {
            "parser": "layout",
            "format": "docx",
            **counts,
            "loader": ("langchain:Docx2txtLoader + python-docx" if lc_error is None else "python-docx"),
            "extra_lines_from_langchain": len(extra),
            "images_ocr": len(ocr_results),
            "notes": "; ".join(notes) if notes else None,
        },
    }


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def _pdf_pages_native(path: Path) -> list:
    import fitz

    document = fitz.open(str(path))
    try:
        return [page.get_text("text").strip() for page in document]
    finally:
        document.close()


def _pdf(path: Path) -> dict:
    pages, used, errors = None, None, []

    for name in ("PyMuPDFLoader", "PyPDFLoader"):
        loader = _langchain_loader(name)
        if loader is None:
            errors.append(f"{name} unavailable")
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                docs = loader(str(path)).load()
            pages, used = [(d.page_content or "").strip() for d in docs], f"langchain:{name}"
            break
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    if pages is None:
        pages, used = _pdf_pages_native(path), "pymupdf"   # a corrupt PDF raises here, as before

    components, blocks = [], []
    for number, text in enumerate(pages, start=1):
        if text:
            blocks.append(_escape_lines(text))
            components.append({"type": "page", "page": number, "text": text})

    return {
        "text": "\n\n".join(blocks),
        "components": components,
        "metadata": {
            "parser": "layout",
            "format": "pdf",
            "page_count": len(components),
            "total_pages": len(pages),
            "loader": used,
            "notes": "; ".join(errors) if errors else None,
        },
    }


# ---------------------------------------------------------------------------
# HTML  (unchanged behaviour)
# ---------------------------------------------------------------------------
def _html(path: Path) -> dict:
    from bs4 import BeautifulSoup

    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    for node in soup(["script", "style", "noscript", "template"]):
        node.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else None
    text = soup.get_text("\n", strip=True)
    components = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "code", "blockquote"]):
        value = tag.get_text(" ", strip=True)
        if value:
            components.append({"type": tag.name, "text": value})

    return {
        "text": text,
        "components": components,
        "metadata": {"parser": "layout", "format": "html", "title": title},
    }


# ---------------------------------------------------------------------------
# entry point (unchanged signature)
# ---------------------------------------------------------------------------
def parse(file_path: str | Path) -> dict:
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext == ".docx":
        return _docx(path)
    if ext == ".pdf":
        return _pdf(path)
    if ext in {".html", ".htm"}:
        return _html(path)
    raise ValueError(f"layout parser does not support {ext!r}")