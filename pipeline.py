"""Local RAG pipeline orchestrator.

This module deliberately stays format-agnostic. Stage 1 owns file parsing and
routing; this module owns lifecycle, synchronization, and stage-to-stage flow.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from src.stage_1_ingestion.router import ingestion_routing_validation
from src.stage_1_ingestion.utils.pdf_detector import has_text_layer
from src.stage_2_chunking.chunker import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOKENIZER,
    recursive_structure_aware_chunking,
)
from src.stage_3_embedding.embedder import DEFAULT_BATCH_SIZE, generate_embeddings
from src.stage_4_retrieval.storage import (
    DEFAULT_STORAGE_ROOT,
    DEFAULT_COLLECTION_NAME,
    _initialize_chroma,
    _initialize_sqlite,
    store_embeddings,
)
from src.stage_5_retrieval.retriever import retrieve
from src.stage_6_generation.generator import (
    DEFAULT_OLLAMA_URL,
    DEFAULT_MAX_PROMPT_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    generate_answer,
    ollama_llm_client,
)

Parser = Callable[[Path], dict]
PDFTextDetector = Callable[[Path], bool]
EmbeddingClient = Callable[[list[str]], list[list[float]]]


def _resolve_parser(module_name: str) -> Parser | None:
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None
    parser = getattr(module, "parse", None)
    return parser if callable(parser) else None


def _normalise_path(path: Path) -> str:
    return str(path.expanduser().resolve(strict=False))


def _sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class RAGPipeline:
    """Application orchestrator for Stages 1-6 and document lifecycle."""

    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        direct_text_parser: Parser | None = None,
        structured_data_parser: Parser | None = None,
        layout_parser: Parser | None = None,
        ocr_parser: Parser | None = None,
        pdf_text_detector: PDFTextDetector | None = None,
        parser_registry: dict[str, Parser] | None = None,
        storage_root: str | Path = DEFAULT_STORAGE_ROOT,
        chroma_directory: str | Path | None = None,
        sqlite_path: str | Path | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        ollama_model: str = "qwen3:4b",
        ollama_url: str = DEFAULT_OLLAMA_URL,
    ) -> None:
        if not callable(embedding_client):
            raise TypeError("embedding_client must be callable.")

        self.embedding_client = embedding_client
        self.direct_text_parser = direct_text_parser or _resolve_parser(
            "src.stage_1_ingestion.parsers.direct_text"
        )
        self.structured_data_parser = structured_data_parser or _resolve_parser(
            "src.stage_1_ingestion.parsers.structured"
        )
        self.layout_parser = layout_parser or _resolve_parser(
            "src.stage_1_ingestion.parsers.layout"
        )
        self.ocr_parser = ocr_parser or _resolve_parser(
            "src.stage_1_ingestion.parsers.ocr"
        )
        self.pdf_text_detector = pdf_text_detector or has_text_layer

        self.parser_registry: dict[str, Parser] = {}
        for key, parser in (parser_registry or {}).items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("parser_registry keys must be non-empty strings.")
            if not callable(parser):
                raise TypeError(f"Parser registered for {key!r} is not callable.")
            self.parser_registry[key.strip().lower()] = parser

        self.storage_root = Path(storage_root)
        self.chroma_directory = Path(chroma_directory) if chroma_directory else self.storage_root / "chroma"
        self.sqlite_path = Path(sqlite_path) if sqlite_path else self.storage_root / "documents.db"
        self.collection_name = collection_name
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url

        self.sqlite_connection: sqlite3.Connection | None = None
        self.chroma_client: Any | None = None
        self.chroma_collection: Any | None = None
        self._closed = False
        self._initialize_resources()
        self._initialize_manifest()

    # ---------------- lifecycle ----------------
    def _initialize_resources(self) -> None:
        self.chroma_directory.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.chroma_client, self.chroma_collection = _initialize_chroma(
                self.chroma_directory, self.collection_name
            )
            self.sqlite_connection = sqlite3.connect(
                str(self.sqlite_path), timeout=60.0
            )
            self.sqlite_connection.execute("PRAGMA foreign_keys = ON")
            self.sqlite_connection.execute("PRAGMA journal_mode = WAL")
            self.sqlite_connection.execute("PRAGMA busy_timeout = 60000")
            _initialize_sqlite(self.sqlite_connection)
        except Exception:
            if self.sqlite_connection is not None:
                self.sqlite_connection.close()
                self.sqlite_connection = None
            self.chroma_client = None
            self.chroma_collection = None
            raise

    def _initialize_manifest(self) -> None:
        assert self.sqlite_connection is not None
        self.sqlite_connection.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_documents (
                document_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL UNIQUE,
                content_hash TEXT NOT NULL,
                chunk_ids_json TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self.sqlite_connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_rag_documents_hash ON rag_documents(content_hash)"
        )
        self.sqlite_connection.commit()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("RAGPipeline is closed.")
        if self.sqlite_connection is None or self.chroma_collection is None:
            raise RuntimeError("RAGPipeline resources are not initialized.")

    def close(self) -> None:
        if self._closed:
            return
        if self.sqlite_connection is not None:
            self.sqlite_connection.close()
        self.sqlite_connection = None
        self.chroma_collection = None
        self.chroma_client = None
        self._closed = True

    def __enter__(self) -> "RAGPipeline":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    # ---------------- Stage 1 ----------------
    def _run_stage1(self, path: Path, source_metadata: dict | None) -> dict:
        registry_parser = self.parser_registry.get(path.suffix.lower())
        if registry_parser is None:
            registry_parser = self.parser_registry.get("*")

        # The registry is forwarded to Stage 1 itself. It is intentionally not
        # emulated by pipeline-side extension logic.
        kwargs = {
            "source_metadata": source_metadata,
            "pdf_text_detector": self.pdf_text_detector,
            "direct_text_parser": registry_parser or self.direct_text_parser,
            "structured_data_parser": self.structured_data_parser,
            "layout_parser": self.layout_parser,
            "ocr_parser": self.ocr_parser,
        }
        # Support both the finalized Stage-1 router (which accepts
        # parser_registry) and an older router while the project is being
        # upgraded. Inspect the callable instead of catching arbitrary
        # TypeErrors raised from inside the router.
        try:
            accepts_registry = "parser_registry" in inspect.signature(
                ingestion_routing_validation
            ).parameters
        except (TypeError, ValueError):
            accepts_registry = False

        if accepts_registry:
            return ingestion_routing_validation(
                path, parser_registry=self.parser_registry, **kwargs
            )
        return ingestion_routing_validation(path, **kwargs)

    # ---------------- ingestion ----------------
    def ingest(
        self,
        file_path: str | Path,
        *,
        source_metadata: dict | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        separators: list | None = None,
        tokenizer_name: str = DEFAULT_TOKENIZER,
        embedding_batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> dict:
        """Force ingestion of one file through Stages 1-4."""
        self._ensure_open()
        path = Path(file_path)
        if not path.exists():
            return {"status": "failed", "failed_stage": "stage_1_ingestion", "error": f"File does not exist: {path}"}
        if not path.is_file():
            return {"status": "failed", "failed_stage": "stage_1_ingestion", "error": f"Path is not a file: {path}"}

        document_id = hashlib.sha256(_normalise_path(path).encode("utf-8")).hexdigest()
        content_hash = _sha256_file(path)
        metadata = dict(source_metadata or {})
        metadata.update({"document_id": document_id, "content_hash": content_hash})

        try:
            stage1 = self._run_stage1(path, metadata)
        except Exception as exc:
            return {"status": "failed", "failed_stage": "stage_1_ingestion", "error": f"{type(exc).__name__}: {exc}"}
        if stage1.get("status") != "processed":
            return {"status": "failed", "failed_stage": "stage_1_ingestion", "stage_1": stage1}

        try:
            stage2 = recursive_structure_aware_chunking(
                stage1,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
                tokenizer_name=tokenizer_name,
            )
        except Exception as exc:
            return {"status": "failed", "failed_stage": "stage_2_chunking", "stage_1": stage1, "error": f"{type(exc).__name__}: {exc}"}
        if stage2.get("status") != "chunked":
            return {"status": "failed", "failed_stage": "stage_2_chunking", "stage_1": stage1, "stage_2": stage2}

        try:
            stage3 = generate_embeddings(stage2, embedding_client=self.embedding_client, batch_size=embedding_batch_size)
        except Exception as exc:
            return {"status": "failed", "failed_stage": "stage_3_embedding", "stage_1": stage1, "stage_2": stage2, "error": f"{type(exc).__name__}: {exc}"}
        if stage3.get("status") != "embedded":
            return {"status": "failed", "failed_stage": "stage_3_embedding", "stage_1": stage1, "stage_2": stage2, "stage_3": stage3}

        old_chunk_ids = self._document_chunk_ids(_normalise_path(path))
        new_chunk_ids = [c["chunk_id"] for c in stage3.get("chunks", [])]

        try:
            stage4 = store_embeddings(
                stage3,
                storage_root=self.storage_root,
                chroma_directory=self.chroma_directory,
                sqlite_path=self.sqlite_path,
                collection_name=self.collection_name,
                sqlite_connection=self.sqlite_connection,
                chroma_collection=self.chroma_collection,
            )
        except Exception as exc:
            return {"status": "failed", "failed_stage": "stage_4_storage", "stage_1": stage1, "stage_2": stage2, "stage_3": stage3, "error": f"{type(exc).__name__}: {exc}"}
        if stage4.get("status") != "stored":
            return {"status": "failed", "failed_stage": "stage_4_storage", "stage_1": stage1, "stage_2": stage2, "stage_3": stage3, "stage_4": stage4}

        stale_ids = sorted(set(old_chunk_ids) - set(new_chunk_ids))
        try:
            self._remove_stale_chunks(stale_ids)
            self._write_manifest(document_id, _normalise_path(path), content_hash, new_chunk_ids)
        except Exception as exc:
            return {
                "status": "failed",
                "failed_stage": "synchronization",
                "stage_1": stage1,
                "stage_2": stage2,
                "stage_3": stage3,
                "stage_4": stage4,
                "error": f"{type(exc).__name__}: {exc}",
            }

        return {
            "status": "success",
            "file": stage1.get("file"),
            "document_id": document_id,
            "content_hash": content_hash,
            "replaced_chunk_count": len(stale_ids),
            "stage_1": stage1,
            "stage_2": stage2,
            "stage_3": stage3,
            "stage_4": stage4,
        }

    def sync_file(self, file_path: str | Path, **ingest_options: Any) -> dict:
        """Ingest only when the file is new or its content hash changed."""
        self._ensure_open()
        path = Path(file_path)
        if not path.is_file():
            return {"status": "failed", "error": f"Path is not a file: {path}"}
        source_path = _normalise_path(path)
        content_hash = _sha256_file(path)
        row = self.sqlite_connection.execute(
            "SELECT document_id, content_hash, chunk_ids_json FROM rag_documents WHERE source_path = ?",
            (source_path,),
        ).fetchone()
        if row and row[1] == content_hash and row[2]:
            return {
                "status": "skipped",
                "reason": "unchanged",
                "file": {"name": path.name, "path": source_path},
                "document_id": row[0],
                "content_hash": content_hash,
                "chunk_count": len(json.loads(row[2])),
            }
        return self.ingest(path, **ingest_options)

    def ingest_directory(self, directory: str | Path, *, recursive: bool = True, **options: Any) -> list[dict]:
        """Backward-compatible bulk ingestion; forces ingestion."""
        root = Path(directory)
        if not root.is_dir():
            raise NotADirectoryError(root)
        iterator = root.rglob("*") if recursive else root.glob("*")
        return [self.ingest(p, **options) for p in sorted(p for p in iterator if p.is_file())]

    def sync_directory(self, directory: str | Path, *, recursive: bool = True, **options: Any) -> list[dict]:
        """Synchronize every file in a directory; unchanged files are skipped."""
        root = Path(directory)
        if not root.is_dir():
            raise NotADirectoryError(root)
        iterator = root.rglob("*") if recursive else root.glob("*")
        results = [self.sync_file(p, **options) for p in sorted(p for p in iterator if p.is_file())]
        return results

    # ---------------- manifest / cleanup ----------------
    def _document_chunk_ids(self, source_path: str) -> list[str]:
        row = self.sqlite_connection.execute(
            "SELECT chunk_ids_json FROM rag_documents WHERE source_path = ?",
            (source_path,),
        ).fetchone()
        if row:
            try:
                value = json.loads(row[0])
                if isinstance(value, list) and all(isinstance(x, str) for x in value):
                    return value
            except json.JSONDecodeError:
                pass

        # Legacy fallback: discover chunks created before the manifest existed.
        ids: list[str] = []
        for chunk_id, metadata_json in self.sqlite_connection.execute("SELECT chunk_id, metadata_json FROM chunks"):
            try:
                metadata = json.loads(metadata_json)
            except Exception:
                continue
            stored_path = metadata.get("file_path")
            if stored_path:
                try:
                    stored_path = _normalise_path(Path(stored_path))
                except (TypeError, ValueError, OSError):
                    pass
            if stored_path == source_path:
                ids.append(chunk_id)
        return ids

    def _remove_stale_chunks(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return

        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self.sqlite_connection.execute(
            f"SELECT chunk_id, text, token_count, metadata_json, embedding_dimension"
            f" FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        if not rows:
            return

        # Capture Chroma state before any deletion so we can restore it if
        # Chroma delete fails after SQLite has already committed.
        # Chroma may return NumPy arrays — never use `or []` on them directly.
        old_vector_data = self.chroma_collection.get(
            ids=chunk_ids,
            include=["embeddings", "metadatas"],
        )
        raw_ids = old_vector_data.get("ids")
        raw_embeddings = old_vector_data.get("embeddings")
        raw_metadatas = old_vector_data.get("metadatas")
        old_ids = [] if raw_ids is None else list(raw_ids)
        old_embeddings = (
            [] if raw_embeddings is None
            else [list(vector) for vector in raw_embeddings]
        )
        old_metadatas = [] if raw_metadatas is None else list(raw_metadatas)

        # Delete SQLite first — it supports atomic rollback.
        # Chroma is deleted second; if that fails we restore SQLite.
        try:
            self.sqlite_connection.execute("BEGIN")
            self.sqlite_connection.execute(
                f"DELETE FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids
            )
            self.sqlite_connection.commit()
        except Exception:
            self.sqlite_connection.rollback()
            raise

        # SQLite committed. Now remove from Chroma.
        try:
            self.chroma_collection.delete(ids=chunk_ids)
        except Exception:
            # Chroma delete failed — restore SQLite to stay consistent.
            try:
                self.sqlite_connection.execute("BEGIN")
                for row in rows:
                    self.sqlite_connection.execute(
                        """
                        INSERT OR REPLACE INTO chunks
                        (chunk_id, text, token_count, metadata_json, embedding_dimension)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        row,
                    )
                self.sqlite_connection.commit()
            except Exception:
                pass  # SQLite restore failed — both stores inconsistent
            raise

    def _write_manifest(self, document_id: str, source_path: str, content_hash: str, chunk_ids: list[str]) -> None:
        self.sqlite_connection.execute(
            """
            INSERT INTO rag_documents(document_id, source_path, content_hash, chunk_ids_json, status, updated_at)
            VALUES (?, ?, ?, ?, 'active', ?)
            ON CONFLICT(source_path) DO UPDATE SET
                document_id=excluded.document_id,
                content_hash=excluded.content_hash,
                chunk_ids_json=excluded.chunk_ids_json,
                status=excluded.status,
                updated_at=excluded.updated_at
            """,
            (document_id, source_path, content_hash, json.dumps(chunk_ids), time.time()),
        )
        self.sqlite_connection.commit()

    # ---------------- retrieval / generation ----------------
    def _effective_top_k(self, top_k: int | None) -> int:
        if top_k is not None:
            if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
                raise ValueError("top_k must be a positive integer or None.")
            return top_k
        return max(int(self.chroma_collection.count()), 1)

    def retrieve(self, query: str, *, top_k: int | None = None, similarity_threshold: float | None = None, where_filter: dict | None = None, expand_context: bool = True) -> dict:
        self._ensure_open()
        return retrieve(
            query,
            embedding_client=self.embedding_client,
            sqlite_connection=self.sqlite_connection,
            chroma_collection=self.chroma_collection,
            top_k=self._effective_top_k(top_k),
            similarity_threshold=similarity_threshold,
            where_filter=where_filter,
            expand_context=expand_context,
        )

    def ask(self, query: str, *, top_k: int | None = None, similarity_threshold: float | None = None, where_filter: dict | None = None, expand_context: bool = True, max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS, temperature: float = DEFAULT_TEMPERATURE, stream: bool = False) -> dict | Iterator[dict]:
        self._ensure_open()
        retrieval = self.retrieve(query, top_k=top_k, similarity_threshold=similarity_threshold, where_filter=where_filter, expand_context=expand_context)
        if retrieval.get("status") == "failed":
            payload = {"status": "failed", "query": query, "answer": "", "sources": [], "generation": {}, "error": retrieval.get("error", "Retrieval failed."), "retrieval": retrieval}
            if stream:
                def failure_stream() -> Iterator[dict]:
                    yield {"type": "metadata", **payload}
                return failure_stream()
            return payload
        return generate_answer(query, retrieval, llm_client=self._ollama_client, max_prompt_tokens=max_prompt_tokens, max_output_tokens=max_output_tokens, temperature=temperature, stream=stream)

    def _ollama_client(self, prompt: str, *, max_output_tokens: int, temperature: float, stream: bool) -> str | Iterator[str]:
        return ollama_llm_client(prompt, model=self.ollama_model, ollama_url=self.ollama_url, max_output_tokens=max_output_tokens, temperature=temperature, stream=stream)


__all__ = ["RAGPipeline"]