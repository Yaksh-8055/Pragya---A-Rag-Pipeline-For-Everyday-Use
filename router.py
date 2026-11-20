"""Stage 1: robust file validation, routing and parsing.

The router is intentionally the only layer that decides how a physical file is
interpreted. Downstream stages receive one stable ingestion package.
"""

from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from .schemas import normalize_parser_result, validate_parser_result
from .utils.encoding import detect_encoding
from .utils.mime import sniff_mime
from .utils.pdf_detector import has_text_layer



Parser = Callable[[Path], dict]



# Common, text-oriented formats handled by built-in parsers. This is a routing
# registry, not a restriction imposed by main.py/pipeline.py.
DIRECT_EXTENSIONS = {".txt", ".md", ".py", ".java", ".c", ".cpp", ".h", ".hpp", ".js", ".ts", ".css", ".sql", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".log", ".rtf", ".xml"}
STRUCTURED_EXTENSIONS = {".csv", ".json"}
LAYOUT_EXTENSIONS = {".docx", ".pdf", ".html", ".htm"}
OFFICE_EXTENSIONS = {".xlsx", ".pptx"}
OCR_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}



def _build_response(*, status: str, file_metadata: dict, route: str | None = None, parser: str | None = None, text: str = "", components: list | None = None, metadata: dict | None = None, routing_log: list | None = None) -> dict:
    return {
        "status": status,
        "routing": {"route": route, "parser": parser},
        "file": file_metadata,
        "content": {"text": text, "components": components or []},
        "metadata": metadata or {},
        "routing_log": routing_log or [],
    }



def _default_parser_for(extension: str) -> tuple[str, Parser] | tuple[None, None]:
    if extension in DIRECT_EXTENSIONS:
        from .parsers.direct_text import parse
        return "direct_text_parser", parse
    if extension in STRUCTURED_EXTENSIONS:
        from .parsers.structured import parse
        return "structured_data_parser", parse
    if extension in LAYOUT_EXTENSIONS:
        from .parsers.layout import parse
        return "layout_parser", parse
    if extension in OFFICE_EXTENSIONS:
        from .parsers.office import parse
        return "office_parser", parse
    if extension in OCR_IMAGE_EXTENSIONS:
        from .parsers.ocr import parse
        return "ocr_parser", parse
    return None, None



def ingestion_routing_validation(file_path, source_metadata=None, *, pdf_text_detector=None, direct_text_parser=None, structured_data_parser=None, layout_parser=None, ocr_parser=None, parser_registry: dict[str, Parser] | None = None):
    path = Path(file_path)
    log: list[dict[str, Any]] = []

    if not path.exists():
        return _build_response(status="rejected", file_metadata={"name": path.name, "path": str(path)}, routing_log=[{"stage": "validation", "status": "failed", "message": "File does not exist."}], metadata=source_metadata)
    if not path.is_file():
        return _build_response(status="rejected", file_metadata={"name": path.name, "path": str(path)}, routing_log=[{"stage": "validation", "status": "failed", "message": "Input path is not a file."}], metadata=source_metadata)

    stat = path.stat()
    extension = path.suffix.lower()
    mime_type = sniff_mime(path)
    file_metadata = {
        "name": path.name,
        "path": str(path),
        "extension": extension,
        "mime_type": mime_type,
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    log.append({"stage": "validation", "status": "passed", "message": "File exists and is a regular file."})

    registry = {k.lower(): v for k, v in (parser_registry or {}).items()}
    parser_name = None
    parser_function = registry.get(extension) or registry.get(mime_type)
    if parser_function is not None:
        parser_name = getattr(parser_function, "__name__", "registered_parser")

    if parser_function is None:
        parser_name, parser_function = _default_parser_for(extension)

    route = None
    encoding_info = None
    text_layer = None

    if extension in DIRECT_EXTENSIONS:
        route = "direct_read"
        try:
            encoding_info = detect_encoding(path)
            log.append({"stage": "encoding_detection", "status": "passed", **encoding_info})
        except Exception as exc:

            return _build_response(status="failed", file_metadata=file_metadata, route=route, parser=parser_name, routing_log=[*log, {"stage": "encoding_detection", "status": "failed", "error_type": type(exc).__name__, "error": str(exc)}], metadata=source_metadata)

    elif extension in STRUCTURED_EXTENSIONS:
        route = "structured_data"
    elif extension == ".pdf":
        detector = pdf_text_detector or has_text_layer
        try:
            text_layer = detector(path)
            if not isinstance(text_layer, bool):
                raise TypeError("PDF text detector must return bool")
            route = "layout_analysis" if text_layer else "ocr"
            if not text_layer:
                from .parsers.ocr import parse as default_ocr
                if parser_function is None or parser_function is not ocr_parser:
                    parser_function = ocr_parser or default_ocr
                    parser_name = "ocr_parser"
            log.append({"stage": "pdf_text_layer_detection", "status": "passed", "text_layer_present": text_layer, "next_route": route})
        except Exception as exc:
            return _build_response(status="failed", file_metadata=file_metadata, routing_log=[*log, {"stage": "pdf_text_layer_detection", "status": "failed", "error_type": type(exc).__name__, "error": str(exc)}], metadata=source_metadata)
    elif extension in LAYOUT_EXTENSIONS:
        route = "layout_analysis"
    elif extension in OFFICE_EXTENSIONS:
        route = "office_document"
    elif extension in OCR_IMAGE_EXTENSIONS:
        route = "ocr"
    elif parser_function is not None:
        route = "registered_parser"
    else:
        return _build_response(status="rejected", file_metadata=file_metadata, routing_log=[*log, {"stage": "triage", "status": "rejected", "message": f"No Stage 1 parser is registered for extension {extension!r} / MIME {mime_type!r}."}], metadata=source_metadata)

    if parser_function is None:
        return _build_response(status="failed", file_metadata=file_metadata, route=route, parser=parser_name, routing_log=[*log, {"stage": "parser_validation", "status": "failed", "message": "A route was selected but no parser is connected."}], metadata=source_metadata)

    try:
        raw_result = parser_function(path)
        valid, error = validate_parser_result(raw_result)
        if not valid:
            raise TypeError(error)
        result = normalize_parser_result(raw_result)
    except Exception as exc:
        log.append({"stage": "parsing", "status": "failed", "parser": parser_name, "error_type": type(exc).__name__, "error": str(exc)})
        return _build_response(status="failed", file_metadata=file_metadata, route=route, parser=parser_name, routing_log=log, metadata={**(source_metadata or {}), "encoding": encoding_info, "pdf_text_layer_present": text_layer})

    metadata = {
        **(source_metadata or {}),
        **result["metadata"],
        "encoding": encoding_info,
        "pdf_text_layer_present": text_layer,
    }
    log.append({"stage": "parsing", "status": "completed", "parser": parser_name})

    return _build_response(status="processed", file_metadata=file_metadata, route=route, parser=parser_name, text=result["text"], components=result["components"], metadata=metadata, routing_log=log)



__all__ = ["ingestion_routing_validation"]
