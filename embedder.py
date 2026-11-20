from typing import Callable, List
from copy import deepcopy
import math


# ============================================================
# Stage 3 — Embedding Configuration
# ============================================================

DEFAULT_BATCH_SIZE = 32


# ============================================================
# Internal Validation Helpers
# ============================================================

def _failed_response(error: str, *, details=None) -> dict:
    """
    Build a standardized Stage 3 failure response.
    """
    response = {
        "status": "failed",
        "error": error,
        "chunks": [],
    }

    if details is not None:
        response["details"] = details

    return response


def _validate_chunk_package(chunk_package: dict) -> tuple[bool, str]:
    """
    Validate the basic Stage 2 package structure before
    any embedding request is made.
    """

    if not isinstance(chunk_package, dict):
        return False, "chunk_package must be a dictionary."

    if chunk_package.get("status") != "chunked":
        return False, "Stage 2 package must have status='chunked'."

    if "chunks" not in chunk_package:
        return False, "Stage 2 package is missing the 'chunks' field."

    chunks = chunk_package["chunks"]

    if not isinstance(chunks, list):
        return False, "Stage 2 'chunks' field must be a list."

    return True, ""


def _validate_chunks(chunks: list) -> tuple[bool, str]:
    """
    Validate every chunk before embedding starts.

    This prevents malformed Stage 2 data from causing
    unexpected errors halfway through the embedding process.
    """

    seen_chunk_ids = set()

    for index, chunk in enumerate(chunks):

        if not isinstance(chunk, dict):
            return False, (
                f"Chunk at index {index} must be a dictionary."
            )

        # ----------------------------------------------------
        # chunk_id
        # ----------------------------------------------------

        if "chunk_id" not in chunk:
            return False, (
                f"Chunk at index {index} is missing 'chunk_id'."
            )

        chunk_id = chunk["chunk_id"]

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            return False, (
                f"Chunk at index {index} has an invalid 'chunk_id'."
            )

        if chunk_id in seen_chunk_ids:
            return False, (
                f"Duplicate chunk_id detected: '{chunk_id}'."
            )

        seen_chunk_ids.add(chunk_id)

        # ----------------------------------------------------
        # text
        # ----------------------------------------------------

        if "text" not in chunk:
            return False, (
                f"Chunk '{chunk_id}' is missing 'text'."
            )

        text = chunk["text"]

        if not isinstance(text, str):
            return False, (
                f"Chunk '{chunk_id}' text must be a string."
            )

        if not text.strip():
            return False, (
                f"Chunk '{chunk_id}' contains empty text."
            )

        # ----------------------------------------------------
        # token_count
        # ----------------------------------------------------

        if "token_count" not in chunk:
            return False, (
                f"Chunk '{chunk_id}' is missing 'token_count'."
            )

        token_count = chunk["token_count"]

        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count < 0
        ):
            return False, (
                f"Chunk '{chunk_id}' has an invalid 'token_count'."
            )

        # ----------------------------------------------------
        # metadata
        # ----------------------------------------------------

        if "metadata" not in chunk:
            return False, (
                f"Chunk '{chunk_id}' is missing 'metadata'."
            )

        if not isinstance(chunk["metadata"], dict):
            return False, (
                f"Chunk '{chunk_id}' metadata must be a dictionary."
            )

    return True, ""


def _validate_embedding_vector(
    vector,
    *,
    chunk_id: str,
    expected_dimension: int | None = None,
) -> tuple[bool, str, int]:
    """
    Validate one embedding vector.

    Returns:
        (is_valid, error_message, dimension)
    """

    if not isinstance(vector, list):
        return (
            False,
            f"Invalid vector for chunk '{chunk_id}': "
            f"embedding must be a list.",
            0,
        )

    if len(vector) == 0:
        return (
            False,
            f"Invalid vector for chunk '{chunk_id}': "
            f"embedding vector is empty.",
            0,
        )

    for position, value in enumerate(vector):

        # bool is technically an int in Python, so explicitly reject it.
        if isinstance(value, bool):
            return (
                False,
                f"Invalid vector for chunk '{chunk_id}': "
                f"vector value at position {position} is boolean.",
                0,
            )

        if not isinstance(value, (int, float)):
            return (
                False,
                f"Invalid vector for chunk '{chunk_id}': "
                f"vector value at position {position} must be numeric.",
                0,
            )

        if not math.isfinite(float(value)):
            return (
                False,
                f"Invalid vector for chunk '{chunk_id}': "
                f"vector value at position {position} is NaN or infinity.",
                0,
            )

    dimension = len(vector)

    if expected_dimension is not None:
        if dimension != expected_dimension:
            return (
                False,
                f"Embedding dimension mismatch for chunk '{chunk_id}': "
                f"expected {expected_dimension}, got {dimension}.",
                dimension,
            )

    return True, "", dimension


# ============================================================
# Main Stage 3 Function
# ============================================================

def generate_embeddings(
    chunk_package: dict,
    *,
    embedding_client: Callable[[List[str]], List[List[float]]],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    """
    Stage 3 — Vector Embedding Generation.

    Input:
        Standardized chunk package from Stage 2.

        embedding_client:
            Callable that accepts:

                List[str]

            and returns:

                List[List[float]]

            The returned vector at position N must correspond
            to the input text at position N.

        batch_size:
            Maximum number of chunks sent to the embedding
            provider in one request.

    Output:
        Standardized embedding package containing the original
        chunks enriched with embedding vectors.

    Guarantees:
        - Validates Stage 2 input before embedding.
        - Preserves chunk ordering.
        - Preserves chunk IDs.
        - Preserves Stage 2 metadata.
        - Processes embeddings in batches.
        - Validates response size for every batch.
        - Validates every vector.
        - Rejects NaN and infinity values.
        - Enforces one consistent vector dimension.
        - Detects duplicate chunk IDs.
        - Detects malformed chunks.
        - Detects final chunk/vector count mismatches.
        - Never returns a partially embedded package as
          status='embedded'.
    """

    # ========================================================
    # 1. Validate Stage 2 Package
    # ========================================================

    valid_package, package_error = _validate_chunk_package(
        chunk_package
    )

    if not valid_package:
        return _failed_response(package_error)

    chunks = chunk_package["chunks"]

    # ========================================================
    # 2. Validate Embedding Client
    # ========================================================

    if embedding_client is None:
        return _failed_response(
            "An embedding_client function must be provided."
        )

    if not callable(embedding_client):
        return _failed_response(
            "embedding_client must be callable."
        )

    # ========================================================
    # 3. Validate Batch Size
    # ========================================================

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        return _failed_response(
            "batch_size must be an integer."
        )

    if batch_size <= 0:
        return _failed_response(
            "batch_size must be greater than zero."
        )

    # ========================================================
    # 4. Validate Individual Chunks
    # ========================================================

    valid_chunks, chunk_error = _validate_chunks(chunks)

    if not valid_chunks:
        return _failed_response(chunk_error)

    # ========================================================
    # 5. Handle Empty Stage 2 Package
    # ========================================================

    if not chunks:
        return {
            "status": "embedded",
            "document": deepcopy(
                chunk_package.get("document", {})
            ),
            "document_metadata": deepcopy(
                chunk_package.get("document_metadata", {})
            ),
            "chunking_config": deepcopy(
                chunk_package.get("chunking", {})
            ),
            "embedding_config": {
                "batch_size": batch_size,
                "total_embedded": 0,
                "vector_dimension": 0,
            },
            "chunks": [],
        }

    # ========================================================
    # 6. Extract Texts
    # ========================================================

    texts_to_embed = [
        chunk["text"]
        for chunk in chunks
    ]

    # ========================================================
    # 7. Batch Embedding
    # ========================================================

    all_embeddings = []

    total_chunks = len(texts_to_embed)

    for batch_start in range(0, total_chunks, batch_size):

        batch_end = min(
            batch_start + batch_size,
            total_chunks,
        )

        batch_texts = texts_to_embed[
            batch_start:batch_end
        ]

        batch_number = (
            batch_start // batch_size
        ) + 1

        batch_chunk_ids = [
            chunks[index]["chunk_id"]
            for index in range(batch_start, batch_end)
        ]

        # ----------------------------------------------------
        # Call embedding provider
        # ----------------------------------------------------

        try:
            batch_vectors = embedding_client(batch_texts)

        except Exception as error:
            return _failed_response(
                (
                    f"Embedding generation failed in batch "
                    f"{batch_number}: "
                    f"{type(error).__name__}: {error}"
                ),
                details={
                    "batch_number": batch_number,
                    "batch_start": batch_start,
                    "batch_size": len(batch_texts),
                    "chunk_ids": batch_chunk_ids,
                },
            )

        # ----------------------------------------------------
        # Validate provider response type
        # ----------------------------------------------------

        if not isinstance(batch_vectors, list):
            return _failed_response(
                (
                    f"Embedding client returned an invalid "
                    f"response type in batch {batch_number}. "
                    f"Expected list, got "
                    f"{type(batch_vectors).__name__}."
                ),
                details={
                    "batch_number": batch_number,
                    "chunk_ids": batch_chunk_ids,
                },
            )

        # ----------------------------------------------------
        # Validate number of returned vectors
        # ----------------------------------------------------

        if len(batch_vectors) != len(batch_texts):
            return _failed_response(
                (
                    f"Embedding client returned an invalid "
                    f"number of vectors in batch {batch_number}. "
                    f"Expected {len(batch_texts)}, "
                    f"got {len(batch_vectors)}."
                ),
                details={
                    "batch_number": batch_number,
                    "expected_vectors": len(batch_texts),
                    "received_vectors": len(batch_vectors),
                    "chunk_ids": batch_chunk_ids,
                },
            )

        # ----------------------------------------------------
        # Validate every vector in this batch
        # ----------------------------------------------------

        batch_dimension = None

        for local_index, vector in enumerate(batch_vectors):

            chunk_id = batch_chunk_ids[local_index]

            valid_vector, vector_error, dimension = (
                _validate_embedding_vector(
                    vector,
                    chunk_id=chunk_id,
                    expected_dimension=None,
                )
            )

            if not valid_vector:
                return _failed_response(
                    (
                        f"Embedding validation failed in "
                        f"batch {batch_number}: "
                        f"{vector_error}"
                    ),
                    details={
                        "batch_number": batch_number,
                        "chunk_id": chunk_id,
                    },
                )

            # First vector establishes the dimension
            # for this batch.
            if batch_dimension is None:
                batch_dimension = dimension

            elif dimension != batch_dimension:
                return _failed_response(
                    (
                        f"Inconsistent vector dimensions "
                        f"inside batch {batch_number}: "
                        f"expected {batch_dimension}, "
                        f"got {dimension} for chunk "
                        f"'{chunk_id}'."
                    ),
                    details={
                        "batch_number": batch_number,
                        "chunk_id": chunk_id,
                        "expected_dimension": batch_dimension,
                        "received_dimension": dimension,
                    },
                )

        # ----------------------------------------------------
        # Validate against previously processed batches
        # ----------------------------------------------------

        if all_embeddings:

            established_dimension = len(
                all_embeddings[0]
            )

            if batch_dimension != established_dimension:
                return _failed_response(
                    (
                        "Embedding dimension changed between "
                        "batches. "
                        f"Expected {established_dimension}, "
                        f"got {batch_dimension} in batch "
                        f"{batch_number}."
                    ),
                    details={
                        "batch_number": batch_number,
                        "expected_dimension": established_dimension,
                        "received_dimension": batch_dimension,
                    },
                )

        # ----------------------------------------------------
        # Store validated vectors
        # ----------------------------------------------------

        all_embeddings.extend(batch_vectors)

    # ========================================================
    # 8. Final Mapping Integrity Check
    # ========================================================

    if len(all_embeddings) != len(chunks):
        return _failed_response(
            (
                "Final embedding count does not match "
                "the number of input chunks. "
                f"Expected {len(chunks)}, "
                f"got {len(all_embeddings)}."
            )
        )

    # ========================================================
    # 9. Determine Final Vector Dimension
    # ========================================================

    if not all_embeddings:
        return _failed_response(
            "No embeddings were generated."
        )

    vector_dimension = len(all_embeddings[0])

    # Final absolute dimension check.
    for index, vector in enumerate(all_embeddings):

        if len(vector) != vector_dimension:
            return _failed_response(
                (
                    "Final embedding validation failed. "
                    f"Chunk '{chunks[index]['chunk_id']}' "
                    f"has dimension {len(vector)}, "
                    f"expected {vector_dimension}."
                )
            )

    # ========================================================
    # 10. Map Vectors Back to Original Chunks
    # ========================================================

    enriched_chunks = []

    for original_chunk, vector in zip(
        chunks,
        all_embeddings,
    ):

        # Deep copy prevents accidental mutation of the
        # original Stage 2 package through shared dictionaries.
        enriched_chunk = {
            "chunk_id": original_chunk["chunk_id"],
            "text": original_chunk["text"],
            "token_count": original_chunk["token_count"],
            "embedding": list(vector),
            "embedding_dimension": vector_dimension,
            "metadata": deepcopy(
                original_chunk["metadata"]
            ),
        }

        enriched_chunks.append(enriched_chunk)

    # ========================================================
    # 11. Final Output Integrity Check
    # ========================================================

    if len(enriched_chunks) != len(chunks):
        return _failed_response(
            (
                "Final chunk mapping failed. "
                f"Expected {len(chunks)} enriched chunks, "
                f"got {len(enriched_chunks)}."
            )
        )

    # ========================================================
    # 12. Final Standardized Stage 3 Package
    # ========================================================

    return {
        "status": "embedded",
        "document": deepcopy(
            chunk_package.get("document", {})
        ),
        "document_metadata": deepcopy(
            chunk_package.get("document_metadata", {})
        ),
        "chunking_config": deepcopy(
            chunk_package.get("chunking", {})
        ),
        "embedding_config": {
            "batch_size": batch_size,
            "total_embedded": len(enriched_chunks),
            "vector_dimension": vector_dimension,
        },
        "chunks": enriched_chunks,
    }