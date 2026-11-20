from __future__ import annotations

from typing import Callable, List, Any
import json
import math
import sqlite3


# ============================================================
# Stage 5 — Retrieval Configuration
# ============================================================

DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.70
DEFAULT_EXPAND_CONTEXT = False

SUPPORTED_DISTANCE_METRICS = {
    "l2",
    "cosine",
}


# ============================================================
# Internal Response Helper
# ============================================================

def _failed_response(
    error: str,
    *,
    details: dict | None = None,
) -> dict:
    """Build a standardized Stage 5 failure response."""
    response = {
        "status": "failed",
        "error": error,
        "retrieved_count": 0,
        "results": [],
    }
    if details is not None:
        response["details"] = details
    return response


# ============================================================
# Query Validation
# ============================================================

def _validate_query(query: Any) -> tuple[bool, str]:
    if not isinstance(query, str):
        return False, "query must be a string."
    if not query.strip():
        return False, "query cannot be empty or whitespace-only."
    return True, ""


# ============================================================
# Search Configuration Validation
# ============================================================

def _validate_search_configuration(
    top_k: Any,
    similarity_threshold: Any,
    expand_context: Any,
) -> tuple[bool, str]:

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        return False, "top_k must be an integer."
    if top_k <= 0:
        return False, "top_k must be greater than zero."

    if similarity_threshold is not None:
        if isinstance(similarity_threshold, bool) or not isinstance(similarity_threshold, (int, float)):
            return False, "similarity_threshold must be a number or None."
        if not math.isfinite(float(similarity_threshold)):
            return False, "similarity_threshold must be finite."
        if not (0.0 <= float(similarity_threshold) <= 1.0):
            return False, "similarity_threshold must be between 0.0 and 1.0."

    if not isinstance(expand_context, bool):
        return False, "expand_context must be a boolean."

    return True, ""


# ============================================================
# Embedding Validation
# ============================================================

def _validate_query_embedding(vector: Any) -> tuple[bool, str, int]:
    if not isinstance(vector, list):
        return False, "Embedding client must return a list of vectors.", 0
    if len(vector) != 1:
        return False, f"Query embedding client must return exactly one vector, got {len(vector)}.", 0

    query_vector = vector[0]

    if not isinstance(query_vector, list):
        return False, "Query embedding must be a list.", 0
    if not query_vector:
        return False, "Query embedding cannot be empty.", 0

    for index, value in enumerate(query_vector):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False, f"Query embedding contains invalid value at position {index}.", 0
        if not math.isfinite(float(value)):
            return False, f"Query embedding contains NaN or infinity at position {index}.", 0

    return True, "", len(query_vector)


# ============================================================
# SQLite Helpers
# ============================================================

def _initialize_sqlite_schema(connection: sqlite3.Connection) -> None:
    cursor = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
    )
    if cursor.fetchone() is None:
        raise RuntimeError("SQLite document store does not contain the required 'chunks' table.")


def _get_stored_embedding_dimension(connection: sqlite3.Connection) -> int | None:
    cursor = connection.execute("SELECT DISTINCT embedding_dimension FROM chunks")
    dimensions = [row[0] for row in cursor.fetchall()]

    if not dimensions:
        return None
    if len(dimensions) != 1:
        raise RuntimeError(f"SQLite contains inconsistent embedding dimensions: {dimensions}")

    dimension = dimensions[0]
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise RuntimeError("SQLite contains an invalid embedding dimension.")

    return dimension


def _hydrate_chunks(connection: sqlite3.Connection, chunk_ids: list[str]) -> dict[str, dict]:
    if not chunk_ids:
        return {}

    unique_ids = list(dict.fromkeys(chunk_ids))
    placeholders = ",".join(["?"] * len(unique_ids))
    
    query = f"""
        SELECT chunk_id, text, token_count, metadata_json, embedding_dimension
        FROM chunks WHERE chunk_id IN ({placeholders});
    """
    
    cursor = connection.execute(query, unique_ids)
    rows = cursor.fetchall()

    hydrated = {}
    for row in rows:
        chunk_id, text, token_count, metadata_json, embedding_dimension = row
        try:
            metadata = json.loads(metadata_json)
        except Exception as error:
            raise RuntimeError(f"Invalid metadata JSON for chunk '{chunk_id}': {error}") from error

        hydrated[chunk_id] = {
            "chunk_id": chunk_id,
            "text": text,
            "token_count": token_count,
            "metadata": metadata,
            "embedding_dimension": embedding_dimension,
        }
    return hydrated


# ============================================================
# Distance / Similarity Conversion
# ============================================================

def _detect_distance_metric(collection: Any) -> str:
    metadata = getattr(collection, "metadata", None)
    if isinstance(metadata, dict):
        configured_metric = metadata.get("hnsw:space")
        if isinstance(configured_metric, str):
            configured_metric = configured_metric.lower()
            if configured_metric in SUPPORTED_DISTANCE_METRICS:
                return configured_metric
    return "l2"


def _distance_to_similarity(distance: Any, metric: str) -> tuple[bool, str, float]:
    if isinstance(distance, bool) or not isinstance(distance, (int, float)):
        return False, "Distance must be numeric.", 0.0
    
    distance = float(distance)
    if not math.isfinite(distance):
        return False, "Distance contains NaN or infinity.", 0.0

    if metric == "cosine":
        similarity = 1.0 - distance
    elif metric == "l2":
        if distance < 0:
            return False, "L2 distance cannot be negative.", 0.0
        similarity = 1.0 / (1.0 + distance)
    else:
        return False, f"Unsupported distance metric: {metric}.", 0.0

    if not math.isfinite(similarity):
        return False, "Calculated similarity is not finite.", 0.0

    return True, "", max(0.0, min(1.0, similarity))


# ============================================================
# Metadata / Graph Helpers
# ============================================================

def _get_neighbor_id(chunk: dict, direction: str) -> str | None:
    metadata = chunk.get("metadata", {})
    if not isinstance(metadata, dict):
        return None
    neighbor_id = metadata.get(f"{direction}_chunk_id")
    if not isinstance(neighbor_id, str) or not neighbor_id.strip():
        return None
    return neighbor_id


def _build_hydrated_result(
    chunk: dict,
    *,
    similarity_score: float,
    distance: float,
    is_expanded_context: bool,
) -> dict:
    return {
        "chunk_id": chunk["chunk_id"],
        "similarity_score": similarity_score,
        "distance": distance,
        "text": chunk["text"],
        "token_count": chunk["token_count"],
        "metadata": chunk["metadata"],
        "is_expanded_context": is_expanded_context,
        "expanded_neighbors": {"previous": None, "next": None},
    }


# ============================================================
# Main Stage 5 Function
# ============================================================

def retrieve(
    query: str,
    *,
    embedding_client: Callable[[List[str]], List[List[float]]],
    sqlite_connection: sqlite3.Connection,
    chroma_collection: Any,
    top_k: int = DEFAULT_TOP_K,
    similarity_threshold: float | None = DEFAULT_SIMILARITY_THRESHOLD,
    where_filter: dict | None = None,
    expand_context: bool = DEFAULT_EXPAND_CONTEXT,
) -> dict:
    """
    Stage 5 — Online Retrieval Engine.
    
    Expects persistent database connections to be injected at runtime 
    to guarantee low-latency processing.
    """

    # 1. Validate Core Inputs
    valid, error = _validate_query(query)
    if not valid: return _failed_response(error)

    valid, error = _validate_search_configuration(top_k, similarity_threshold, expand_context)
    if not valid: return _failed_response(error)

    if embedding_client is None or not callable(embedding_client):
        return _failed_response("embedding_client must be a provided callable.")
    
    if sqlite_connection is None:
        return _failed_response("sqlite_connection must be provided.")
        
    if chroma_collection is None:
        return _failed_response("chroma_collection must be provided.")

    # 2. Generate Query Embedding
    try:
        query_embedding_result = embedding_client([query])
    except Exception as error:
        return _failed_response(f"Query embedding generation failed: {type(error).__name__}: {error}")

    valid, error, query_dimension = _validate_query_embedding(query_embedding_result)
    if not valid: return _failed_response(error)
    query_vector = query_embedding_result[0]

    # 3. Validate SQLite State
    try:
        _initialize_sqlite_schema(sqlite_connection)
        
        cursor = sqlite_connection.execute("SELECT COUNT(*) FROM chunks")
        if cursor.fetchone()[0] == 0:
            return {"status": "no_matches", "query": query, "retrieved_count": 0, "results": [], "reason": "Document store is empty."}

        stored_dimension = _get_stored_embedding_dimension(sqlite_connection)
        if stored_dimension is None:
            return {"status": "no_matches", "query": query, "retrieved_count": 0, "results": [], "reason": "No embedding dimensions stored."}
            
        if query_dimension != stored_dimension:
            return _failed_response(f"Dimension mismatch. Query: {query_dimension}, Stored: {stored_dimension}")
            
    except Exception as error:
        return _failed_response(f"Failed to validate SQLite document store: {type(error).__name__}: {error}")

    # 4. Validate ChromaDB State
    try:
        chroma_count = chroma_collection.count()
        if chroma_count == 0:
            return {"status": "no_matches", "query": query, "retrieved_count": 0, "results": [], "reason": "ChromaDB is empty."}
            
        distance_metric = _detect_distance_metric(chroma_collection)
    except Exception as error:
        return _failed_response(f"Failed to inspect ChromaDB collection: {type(error).__name__}: {error}")

    # 5. Vector Search
    query_arguments = {
        "query_embeddings": [[float(v) for v in query_vector]],
        "n_results": min(top_k, chroma_count),
        "include": ["distances"],
    }
    
    if where_filter is not None:
        if not isinstance(where_filter, dict):
            return _failed_response("where_filter must be a dictionary or None.")
        query_arguments["where"] = where_filter

    try:
        search_result = chroma_collection.query(**query_arguments)
    except Exception as error:
        return _failed_response(f"ChromaDB vector search failed: {type(error).__name__}: {error}")

    # 6. Parse Search Results
    if not isinstance(search_result, dict):
        return _failed_response("ChromaDB returned an invalid search response.")

    ids_groups = search_result.get("ids")
    distances_groups = search_result.get("distances")

    if not isinstance(ids_groups, list) or not ids_groups:
        return {"status": "no_matches", "query": query, "retrieved_count": 0, "results": [], "reason": "No matching IDs."}
    if not isinstance(distances_groups, list) or not distances_groups:
        return _failed_response("ChromaDB did not return distances.")

    result_ids, result_distances = ids_groups[0], distances_groups[0]
    if len(result_ids) != len(result_distances):
        return _failed_response("ChromaDB returned mismatched IDs and distances.")

    # 7. Convert Distances & Apply Threshold
    candidate_results = []
    for chunk_id, distance in zip(result_ids, result_distances):
        valid, error, similarity = _distance_to_similarity(distance, distance_metric)
        if not valid: return _failed_response(f"Invalid distance for chunk '{chunk_id}': {error}")

        if similarity_threshold is not None and similarity < similarity_threshold:
            continue

        candidate_results.append({
            "chunk_id": chunk_id,
            "distance": float(distance),
            "similarity_score": similarity,
        })

    if not candidate_results:
        return {"status": "no_matches", "query": query, "retrieved_count": 0, "results": [], "reason": "No vectors met the threshold."}

    # 8. Batch Hydrate from SQLite
    candidate_ids = [res["chunk_id"] for res in candidate_results]
    try:
        hydrated_chunks = _hydrate_chunks(sqlite_connection, candidate_ids)
    except Exception as error:
        return _failed_response(f"SQLite chunk hydration failed: {type(error).__name__}: {error}")

    missing_ids = [cid for cid in candidate_ids if cid not in hydrated_chunks]
    if missing_ids:
        return _failed_response("ChromaDB returned chunk IDs missing from SQLite.", details={"missing": missing_ids})

    # 9. Build Base Results
    results = []
    for candidate in candidate_results:
        chunk = hydrated_chunks[candidate["chunk_id"]]
        results.append(_build_hydrated_result(
            chunk,
            similarity_score=candidate["similarity_score"],
            distance=candidate["distance"],
            is_expanded_context=False,
        ))

    # 10. Optional Graph Expansion
    if expand_context:
        neighbor_ids = []
        for result in results:
            if prev_id := _get_neighbor_id(result, "previous"): neighbor_ids.append(prev_id)
            if next_id := _get_neighbor_id(result, "next"): neighbor_ids.append(next_id)

        direct_ids = {res["chunk_id"] for res in results}
        neighbor_ids = [nid for nid in dict.fromkeys(neighbor_ids) if nid not in direct_ids]

        neighbor_chunks = {}
        if neighbor_ids:
            try:
                neighbor_chunks = _hydrate_chunks(sqlite_connection, neighbor_ids)
            except Exception as error:
                return _failed_response(f"SQLite graph expansion failed: {error}")

        for result in results:
            if (prev_id := _get_neighbor_id(result, "previous")) and prev_id in neighbor_chunks:
                prev_chunk = neighbor_chunks[prev_id]
                result["expanded_neighbors"]["previous"] = {
                    "chunk_id": prev_chunk["chunk_id"],
                    "text": prev_chunk["text"],
                    "token_count": prev_chunk["token_count"],
                    "metadata": prev_chunk["metadata"],
                    "is_expanded_context": True,
                }
            if (next_id := _get_neighbor_id(result, "next")) and next_id in neighbor_chunks:
                next_chunk = neighbor_chunks[next_id]
                result["expanded_neighbors"]["next"] = {
                    "chunk_id": next_chunk["chunk_id"],
                    "text": next_chunk["text"],
                    "token_count": next_chunk["token_count"],
                    "metadata": next_chunk["metadata"],
                    "is_expanded_context": True,
                }

    return {
        "status": "success",
        "query": query,
        "retrieved_count": len(results),
        "similarity_threshold": similarity_threshold,
        "distance_metric": distance_metric,
        "top_k": top_k,
        "results": results,
    }