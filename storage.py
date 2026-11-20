from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math
import sqlite3


# ============================================================
# Stage 4 — Local Storage Configuration
# ============================================================

DEFAULT_STORAGE_ROOT = Path("data/storage")

DEFAULT_CHROMA_DIR = DEFAULT_STORAGE_ROOT / "chroma"
DEFAULT_SQLITE_PATH = DEFAULT_STORAGE_ROOT / "documents.db"

DEFAULT_COLLECTION_NAME = "rag_chunks"


# ============================================================
# Internal Helpers
# ============================================================

def _failed_response(
    error: str,
    *,
    details: dict | None = None,
) -> dict:
    """
    Build a standardized Stage 4 failure response.
    """

    response = {
        "status": "failed",
        "error": error,
        "stored_chunks": 0,
    }

    if details is not None:
        response["details"] = details

    return response


def _validate_stage3_package(
    embedding_package: dict,
) -> tuple[bool, str]:

    if not isinstance(embedding_package, dict):
        return False, (
            "embedding_package must be a dictionary."
        )

    if embedding_package.get("status") != "embedded":
        return False, (
            "Stage 3 package must have "
            "status='embedded'."
        )

    if "chunks" not in embedding_package:
        return False, (
            "Stage 3 package is missing 'chunks'."
        )

    if not isinstance(
        embedding_package["chunks"],
        list,
    ):
        return False, (
            "Stage 3 'chunks' must be a list."
        )

    return True, ""


def _validate_chunk(
    chunk: dict,
    index: int,
) -> tuple[bool, str]:

    if not isinstance(chunk, dict):
        return False, (
            f"Chunk at index {index} must be a dictionary."
        )

    # --------------------------------------------------------
    # chunk_id
    # --------------------------------------------------------

    if "chunk_id" not in chunk:
        return False, (
            f"Chunk at index {index} is missing "
            "'chunk_id'."
        )

    chunk_id = chunk["chunk_id"]

    if not isinstance(chunk_id, str):
        return False, (
            f"Chunk at index {index} has an invalid "
            "'chunk_id'."
        )

    if not chunk_id.strip():
        return False, (
            f"Chunk at index {index} has an empty "
            "'chunk_id'."
        )

    # --------------------------------------------------------
    # text
    # --------------------------------------------------------

    if "text" not in chunk:
        return False, (
            f"Chunk '{chunk_id}' is missing 'text'."
        )

    if not isinstance(chunk["text"], str):
        return False, (
            f"Chunk '{chunk_id}' text must be a string."
        )

    if not chunk["text"].strip():
        return False, (
            f"Chunk '{chunk_id}' contains empty text."
        )

    # --------------------------------------------------------
    # token_count
    # --------------------------------------------------------

    if "token_count" not in chunk:
        return False, (
            f"Chunk '{chunk_id}' is missing "
            "'token_count'."
        )

    token_count = chunk["token_count"]

    if (
        isinstance(token_count, bool)
        or not isinstance(token_count, int)
        or token_count < 0
    ):
        return False, (
            f"Chunk '{chunk_id}' has an invalid "
            "'token_count'."
        )

    # --------------------------------------------------------
    # embedding
    # --------------------------------------------------------

    if "embedding" not in chunk:
        return False, (
            f"Chunk '{chunk_id}' is missing "
            "'embedding'."
        )

    embedding = chunk["embedding"]

    if not isinstance(embedding, list):
        return False, (
            f"Embedding for chunk '{chunk_id}' "
            "must be a list."
        )

    if not embedding:
        return False, (
            f"Embedding for chunk '{chunk_id}' "
            "cannot be empty."
        )

    for position, value in enumerate(embedding):

        if isinstance(value, bool):
            return False, (
                f"Embedding for chunk '{chunk_id}' "
                f"contains boolean at position {position}."
            )

        if not isinstance(value, (int, float)):
            return False, (
                f"Embedding for chunk '{chunk_id}' "
                f"contains a non-numeric value at "
                f"position {position}."
            )

        if not math.isfinite(float(value)):
            return False, (
                f"Embedding for chunk '{chunk_id}' "
                f"contains NaN or infinity at "
                f"position {position}."
            )

    # --------------------------------------------------------
    # embedding_dimension
    # --------------------------------------------------------

    if "embedding_dimension" in chunk:

        dimension = chunk["embedding_dimension"]

        if dimension != len(embedding):
            return False, (
                f"Embedding dimension mismatch for "
                f"chunk '{chunk_id}'. "
                f"Declared {dimension}, actual "
                f"{len(embedding)}."
            )

    # --------------------------------------------------------
    # metadata
    # --------------------------------------------------------

    if "metadata" not in chunk:
        return False, (
            f"Chunk '{chunk_id}' is missing "
            "'metadata'."
        )

    if not isinstance(chunk["metadata"], dict):
        return False, (
            f"Metadata for chunk '{chunk_id}' "
            "must be a dictionary."
        )

    return True, ""


def _validate_all_chunks(
    chunks: list,
) -> tuple[bool, str, int]:

    if not chunks:
        return True, "", 0

    seen_ids = set()
    expected_dimension = None

    for index, chunk in enumerate(chunks):

        valid, error = _validate_chunk(
            chunk,
            index,
        )

        if not valid:
            return False, error, 0

        chunk_id = chunk["chunk_id"]

        if chunk_id in seen_ids:
            return False, (
                f"Duplicate chunk_id detected: "
                f"'{chunk_id}'."
            ), 0

        seen_ids.add(chunk_id)

        current_dimension = len(
            chunk["embedding"]
        )

        if expected_dimension is None:
            expected_dimension = current_dimension

        elif current_dimension != expected_dimension:
            return False, (
                "Embedding dimension mismatch across "
                "chunks. "
                f"Expected {expected_dimension}, "
                f"got {current_dimension} for "
                f"chunk '{chunk_id}'."
            ), 0

    return True, "", expected_dimension


def _safe_json(value: Any) -> str:
    """
    Serialize Python metadata safely for SQLite.
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _extract_vector_metadata(
    chunk: dict,
) -> dict:
    """
    Extract only metadata that is useful for
    vector filtering/search.

    Full metadata remains in SQLite.
    """

    metadata = chunk.get("metadata", {})

    result = {
        "chunk_id": chunk["chunk_id"],
    }

    # --------------------------------------------------------
    # File information
    # --------------------------------------------------------

    file_name = metadata.get("file_name")

    if file_name is None:
        file_name = metadata.get("name")

    if file_name is not None:
        result["file_name"] = str(file_name)

    file_extension = metadata.get(
        "file_extension"
    )

    if file_extension is None:
        file_extension = metadata.get(
            "extension"
        )

    if file_extension is not None:
        result["file_extension"] = str(
            file_extension
        )

    # --------------------------------------------------------
    # Section headers
    # --------------------------------------------------------

    section_headers = metadata.get(
        "section_headers",
        [],
    )

    if isinstance(section_headers, list):
        result["section_headers"] = [
            str(header)
            for header in section_headers
        ]

    return result


# ============================================================
# SQLite Initialization
# ============================================================

def _initialize_sqlite(
    connection: sqlite3.Connection,
) -> None:
    """
    Create the local document store schema.
    """

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            metadata_json TEXT NOT NULL,
            embedding_dimension INTEGER NOT NULL
        )
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_chunks_chunk_id
        ON chunks(chunk_id)
        """
    )

    connection.commit()


# ============================================================
# ChromaDB Initialization
# ============================================================

def _initialize_chroma(
    chroma_directory: Path,
    collection_name: str,
):
    """
    Initialize the local persistent ChromaDB collection.
    """

    try:
        import chromadb
    except ImportError as error:
        raise RuntimeError(
            "ChromaDB is not installed. "
            "Install it with: pip install chromadb"
        ) from error

    chroma_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(chroma_directory)
    )

    collection = client.get_or_create_collection(
        name=collection_name
    )

    return client, collection


# ============================================================
# Main Stage 4 Function
# ============================================================

def store_embeddings(
    embedding_package: dict,
    *,
    storage_root: str | Path = DEFAULT_STORAGE_ROOT,
    chroma_directory: str | Path | None = None,
    sqlite_path: str | Path | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    sqlite_connection: sqlite3.Connection | None = None,
    chroma_collection: Any | None = None,
) -> dict:
    """Store Stage-3 embeddings in the local SQLite + Chroma backends.

    The pipeline may inject its already-open SQLite connection and Chroma
    collection. When they are not supplied, Stage 4 keeps its original
    standalone behavior and initializes/closes its own resources.

    SQLite and Chroma do not share an ACID transaction. Therefore writes use
    explicit validation and compensating rollback where a later backend write
    fails.
    """
    valid, error = _validate_stage3_package(embedding_package)
    if not valid:
        return _failed_response(error)

    chunks = embedding_package["chunks"]

    if not isinstance(collection_name, str) or not collection_name.strip():
        return _failed_response("collection_name must be a non-empty string.")

    valid, error, vector_dimension = _validate_all_chunks(chunks)
    if not valid:
        return _failed_response(error)

    root = Path(storage_root)
    chroma_path = Path(chroma_directory) if chroma_directory is not None else root / "chroma"
    database_path = Path(sqlite_path) if sqlite_path is not None else root / "documents.db"

    try:
        chroma_path.mkdir(parents=True, exist_ok=True)
        database_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return _failed_response(f"Failed to create local storage directories: {exc}")

    owns_connection = sqlite_connection is None
    owns_collection = chroma_collection is None
    connection = sqlite_connection
    collection = chroma_collection

    try:
        if collection is None:
            _, collection = _initialize_chroma(chroma_path, collection_name)
        if connection is None:
            connection = sqlite3.connect(str(database_path), timeout=60.0)
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 60000")
            _initialize_sqlite(connection)
        # When the pipeline injects its connection it has already initialized
        # the schema in _initialize_resources — skip the redundant call.
    except Exception as exc:
        if owns_connection and connection is not None:
            connection.close()
        return _failed_response(
            f"Failed to initialize local storage: {type(exc).__name__}: {exc}"
        )

    try:
        if not chunks:
            return {
                "status": "stored",
                "document": embedding_package.get("document", {}),
                "document_metadata": embedding_package.get("document_metadata", {}),
                "storage": {
                    "backend": "local",
                    "vector_store": "chromadb",
                    "document_store": "sqlite",
                    "collection_name": collection_name,
                    "chroma_path": str(chroma_path),
                    "sqlite_path": str(database_path),
                    "stored_chunks": 0,
                    "vector_dimension": 0,
                },
            }

        chroma_ids = [chunk["chunk_id"] for chunk in chunks]
        chroma_embeddings = [
            [float(value) for value in chunk["embedding"]]
            for chunk in chunks
        ]
        chroma_metadatas = []
        for chunk in chunks:
            metadata = _extract_vector_metadata(chunk)
            if "section_headers" in metadata:
                metadata["section_headers"] = _safe_json(metadata["section_headers"])
            chroma_metadatas.append(metadata)

        # Capture existing rows/vectors for these IDs so a Chroma failure can
        # restore the previous representation rather than leaving a half-new
        # document behind.
        placeholders = ",".join("?" for _ in chroma_ids)
        old_rows = connection.execute(
            f"SELECT chunk_id, text, token_count, metadata_json, embedding_dimension "
            f"FROM chunks WHERE chunk_id IN ({placeholders})",
            chroma_ids,
        ).fetchall()
        old_vector_data = collection.get(
            ids=chroma_ids,
            include=["embeddings", "metadatas"],
        )

        # Chroma may return NumPy arrays here, so never use `or []`
        # or direct truth-value checks on the returned embeddings.
        old_ids = old_vector_data.get("ids")
        old_embeddings = old_vector_data.get("embeddings")
        old_metadatas = old_vector_data.get("metadatas")

        old_ids = [] if old_ids is None else list(old_ids)
        old_embeddings = (
            [] if old_embeddings is None
            else [list(vector) for vector in old_embeddings]
        )
        old_metadatas = (
            [] if old_metadatas is None
            else list(old_metadatas)
        )

        connection.execute("BEGIN")
        for chunk in chunks:
            connection.execute(
                """
                INSERT OR REPLACE INTO chunks (
                    chunk_id, text, token_count, metadata_json, embedding_dimension
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    chunk["chunk_id"],
                    chunk["text"],
                    chunk["token_count"],
                    _safe_json(chunk["metadata"]),
                    vector_dimension,
                ),
            )
        connection.commit()

        try:
            collection.upsert(
                ids=chroma_ids,
                embeddings=chroma_embeddings,
                metadatas=chroma_metadatas,
            )
        except Exception as exc:
            connection.execute("BEGIN")
            connection.execute(
                f"DELETE FROM chunks WHERE chunk_id IN ({placeholders})",
                chroma_ids,
            )
            for row in old_rows:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO chunks
                    (chunk_id, text, token_count, metadata_json, embedding_dimension)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    row,
                )
            connection.commit()

            collection.delete(ids=chroma_ids)
            if len(old_ids) > 0 and len(old_embeddings) > 0:
                restore = {"ids": old_ids, "embeddings": old_embeddings}
                if old_metadatas:
                    restore["metadatas"] = old_metadatas
                collection.upsert(**restore)

            return _failed_response(
                f"SQLite storage was rolled back because ChromaDB vector storage failed: "
                f"{type(exc).__name__}: {exc}",
                details={
                    "sqlite_path": str(database_path),
                    "chroma_path": str(chroma_path),
                    "collection_name": collection_name,
                },
            )

        try:
            chroma_count = collection.count()
        except Exception as exc:
            return _failed_response(
                f"Vectors were written to ChromaDB, but collection verification failed: "
                f"{type(exc).__name__}: {exc}"
            )

        return {
            "status": "stored",
            "document": embedding_package.get("document", {}),
            "document_metadata": embedding_package.get("document_metadata", {}),
            "storage": {
                "backend": "local",
                "vector_store": "chromadb",
                "document_store": "sqlite",
                "collection_name": collection_name,
                "chroma_path": str(chroma_path),
                "sqlite_path": str(database_path),
                "stored_chunks": len(chunks),
                "vector_dimension": vector_dimension,
                "chroma_collection_count": chroma_count,
            },
        }
    except Exception as exc:
        try:
            connection.rollback()
        except Exception:
            pass
        return _failed_response(
            f"Failed to store embeddings: {type(exc).__name__}: {exc}"
        )
    finally:
        if owns_connection and connection is not None:
            try:
                connection.close()
            except Exception:
                pass