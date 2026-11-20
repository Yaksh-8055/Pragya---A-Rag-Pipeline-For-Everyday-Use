import re
from typing import Callable, Optional

import tiktoken


# ============================================================
# Stage 2 — Chunking Configuration
# ============================================================

DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 70

DEFAULT_TOKENIZER = "cl100k_base"

DEFAULT_SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
]


# ============================================================
# Tokenizer
# ============================================================

def _build_token_counter(
    tokenizer_name: str = DEFAULT_TOKENIZER,
) -> Callable[[str], int]:
    """
    Build a tiktoken-based token counter.

    The tokenizer is loaded once and then reused.
    """

    encoding = tiktoken.get_encoding(tokenizer_name)

    def token_counter(text: str) -> int:
        if not text:
            return 0

        return len(encoding.encode(text))

    return token_counter


# ============================================================
# Text Normalization
# ============================================================

def _normalize_text(text: str) -> str:
    """
    Normalize line endings and excessive blank lines while
    preserving meaningful document structure.
    """

    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = "\n".join(
        line.rstrip()
        for line in text.split("\n")
    )

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# Markdown Header Detection
# ============================================================

MARKDOWN_HEADER_PATTERN = re.compile(
    r"^(#{1,6})\s+(.+?)\s*$"
)


def _split_markdown_sections(text: str) -> list:
    """
    Split Markdown into structural sections.

    Each section keeps its own header hierarchy.

    Overlap is later applied only inside the same section.
    """

    lines = text.splitlines()

    sections = []

    current_lines = []
    current_headers = []

    for line in lines:

        match = MARKDOWN_HEADER_PATTERN.match(line)

        if match:

            if current_lines:

                section_text = "\n".join(
                    current_lines
                ).strip()

                if section_text:
                    sections.append({
                        "text": section_text,
                        "headers": list(current_headers),
                    })

            level = len(match.group(1))
            header = match.group(2).strip()

            current_headers = [
                item
                for item in current_headers
                if item["level"] < level
            ]

            current_headers.append({
                "level": level,
                "header": header,
            })

            current_lines = [line]

        else:
            current_lines.append(line)

    if current_lines:

        section_text = "\n".join(
            current_lines
        ).strip()

        if section_text:
            sections.append({
                "text": section_text,
                "headers": list(current_headers),
            })

    return sections


# ============================================================
# Hard Split
# ============================================================

def _hard_split(
    text: str,
    chunk_size: int,
    token_counter: Callable[[str], int],
) -> list:
    """
    Final safety fallback when no useful separator remains.

    Uses incremental token accounting rather than repeatedly
    rebuilding and tokenizing the complete candidate.

    The completed chunks are checked once before returning so
    the token limit remains a hard limit.
    """

    words = text.split()

    if not words:
        return []

    chunks = []

    current_words = []
    current_token_count = 0

    for word in words:

        if not current_words:

            current_words = [word]
            current_token_count = token_counter(word)

            # A single word can itself exceed the target.
            # It cannot be split further at the whitespace level.
            if current_token_count > chunk_size:

                chunks.append(word)
                current_words = []
                current_token_count = 0

            continue

        candidate_words = current_words + [word]

        # Fast approximate accounting.
        word_token_count = token_counter(
            " " + word
        )

        estimated_count = (
            current_token_count
            + word_token_count
        )

        if estimated_count <= chunk_size:

            current_words.append(word)
            current_token_count = estimated_count

        else:

            current_chunk = " ".join(current_words)

            # Exact final validation.
            actual_count = token_counter(
                current_chunk
            )

            if actual_count > chunk_size:

                # This should be extremely rare because the
                # incremental accounting normally catches it.
                chunks.extend(
                    _exact_token_safe_split(
                        current_chunk,
                        chunk_size,
                        token_counter,
                    )
                )

            else:
                chunks.append(current_chunk)

            current_words = [word]
            current_token_count = token_counter(word)

            if current_token_count > chunk_size:

                chunks.append(word)
                current_words = []
                current_token_count = 0

    if current_words:

        current_chunk = " ".join(current_words)

        actual_count = token_counter(
            current_chunk
        )

        if actual_count <= chunk_size:

            chunks.append(current_chunk)

        else:

            chunks.extend(
                _exact_token_safe_split(
                    current_chunk,
                    chunk_size,
                    token_counter,
                )
            )

    return chunks


# ============================================================
# Exact Token-Safe Fallback
# ============================================================

def _exact_token_safe_split(
    text: str,
    chunk_size: int,
    token_counter: Callable[[str], int],
) -> list:
    """
    Exact safety fallback for text whose incremental token
    accounting crossed the configured limit.

    This performs a bounded binary search for the largest
    whitespace-delimited prefix that fits.
    """

    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):

        low = start + 1
        high = len(words)
        best = start

        while low <= high:

            middle = (low + high) // 2

            candidate = " ".join(
                words[start:middle]
            )

            count = token_counter(candidate)

            if count <= chunk_size:

                best = middle
                low = middle + 1

            else:

                high = middle - 1

        if best == start:

            # One whitespace-delimited token/string is larger
            # than the requested limit and cannot be split safely
            # at the current textual boundary.
            chunks.append(words[start])
            start += 1

        else:

            chunks.append(
                " ".join(words[start:best])
            )

            start = best

    return chunks


# ============================================================
# Recursive Split
# ============================================================

def _recursive_split(
    text: str,
    chunk_size: int,
    separators: list,
    token_counter: Callable[[str], int],
) -> list:
    """
    Recursively split text according to separator priority.

    No returned piece exceeds chunk_size.
    """

    text = text.strip()

    if not text:
        return []

    if token_counter(text) <= chunk_size:
        return [text]

    if not separators:

        return _hard_split(
            text,
            chunk_size,
            token_counter,
        )

    separator = separators[0]

    if separator == "":

        return _hard_split(
            text,
            chunk_size,
            token_counter,
        )

    parts = text.split(separator)

    if len(parts) == 1:

        return _recursive_split(
            text,
            chunk_size,
            separators[1:],
            token_counter,
        )

    pieces = []

    for part in parts:

        part = part.strip()

        if not part:
            continue

        if token_counter(part) <= chunk_size:

            pieces.append(part)

        else:

            pieces.extend(
                _recursive_split(
                    part,
                    chunk_size,
                    separators[1:],
                    token_counter,
                )
            )

    return _merge_small_pieces(
        pieces=pieces,
        separator=separator,
        chunk_size=chunk_size,
        token_counter=token_counter,
    )


# ============================================================
# Merge Small Pieces
# ============================================================

def _merge_small_pieces(
    pieces: list,
    separator: str,
    chunk_size: int,
    token_counter: Callable[[str], int],
) -> list:
    """
    Merge neighboring pieces without exceeding chunk_size.

    This stage performs no overlap.

    Overlap is introduced later during construction of the
    next chunk, where the resulting chunk is still checked
    against the hard token limit.
    """

    merged = []
    current = ""

    for piece in pieces:

        if not current:

            current = piece
            continue

        candidate = (
            current
            + separator
            + piece
        )

        if token_counter(candidate) <= chunk_size:

            current = candidate

        else:

            merged.append(current.strip())
            current = piece

    if current.strip():
        merged.append(current.strip())

    return merged


# ============================================================
# Extract Overlap Text
# ============================================================

def _get_overlap_text(
    previous_chunk: str,
    overlap_tokens: int,
    token_counter: Callable[[str], int],
) -> str:
    """
    Extract approximately overlap_tokens worth of trailing
    context from the previous chunk.

    The overlap is taken from the end of the previous chunk.
    """

    if overlap_tokens <= 0:
        return ""

    words = previous_chunk.split()

    if not words:
        return ""

    selected = []

    for word in reversed(words):

        candidate = " ".join(
            [word] + selected
        )

        if token_counter(candidate) <= overlap_tokens:

            selected.insert(0, word)

        else:

            break

    return " ".join(selected)


# ============================================================
# Construct Chunks With Safe Overlap
# ============================================================

def _construct_chunks_with_overlap(
    pieces: list,
    chunk_size: int,
    overlap_tokens: int,
    token_counter: Callable[[str], int],
) -> list:
    """
    Construct final chunks from recursively split pieces.

    Important:
        Overlap is introduced while creating the next chunk,
        rather than after all chunks have already been created.

    The overlap is reduced whenever necessary so the final
    chunk NEVER exceeds chunk_size.
    """

    if not pieces:
        return []

    chunks = []

    current = pieces[0].strip()

    for piece in pieces[1:]:

        piece = piece.strip()

        if not piece:
            continue

        candidate = (
            current
            + "\n"
            + piece
        )

        if token_counter(candidate) <= chunk_size:

            current = candidate
            continue

        # ----------------------------------------------------
        # Current chunk is complete.
        # ----------------------------------------------------

        chunks.append(current.strip())

        # ----------------------------------------------------
        # Build overlap for the next chunk.
        # ----------------------------------------------------

        overlap_text = _get_overlap_text(
            previous_chunk=current,
            overlap_tokens=overlap_tokens,
            token_counter=token_counter,
        )

        next_chunk = piece

        if overlap_text:

            candidate = (
                overlap_text
                + "\n"
                + piece
            )

            # If the full requested overlap does not fit,
            # progressively reduce it.
            if token_counter(candidate) <= chunk_size:

                next_chunk = candidate

            else:

                overlap_words = overlap_text.split()

                while overlap_words:

                    overlap_words.pop(0)

                    if not overlap_words:
                        break

                    reduced_overlap = " ".join(
                        overlap_words
                    )

                    candidate = (
                        reduced_overlap
                        + "\n"
                        + piece
                    )

                    if token_counter(candidate) <= chunk_size:

                        next_chunk = candidate
                        break

        current = next_chunk

    # --------------------------------------------------------
    # Final chunk
    # --------------------------------------------------------

    if current.strip():
        chunks.append(current.strip())

    # --------------------------------------------------------
    # Hard validation
    # --------------------------------------------------------

    validated_chunks = []

    for chunk in chunks:

        token_count = token_counter(chunk)

        if token_count <= chunk_size:

            validated_chunks.append(chunk)

        else:

            # Absolute safety net.
            validated_chunks.extend(
                _exact_token_safe_split(
                    chunk,
                    chunk_size,
                    token_counter,
                )
            )

    return validated_chunks


# ============================================================
# Chunk Metadata
# ============================================================

def _build_chunk_metadata(
    ingestion_package: dict,
    section_headers: Optional[list],
    chunk_index: int,
) -> dict:
    """
    Build metadata inherited from Stage 1.
    """

    file_metadata = ingestion_package.get(
        "file",
        {},
    )

    document_metadata = ingestion_package.get(
        "metadata",
        {},
    )

    file_name = file_metadata.get(
        "name",
        "document",
    )

    safe_name = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        file_name,
    ).strip("_")

    if not safe_name:
        safe_name = "document"

    chunk_id = (
        f"{safe_name}_chunk_{chunk_index:03d}"
    )

    return {
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,

        "file_name": file_metadata.get("name"),
        "file_path": file_metadata.get("path"),
        "file_extension": file_metadata.get("extension"),
        "file_size_bytes": file_metadata.get("size_bytes"),

        "document_metadata": document_metadata,

        "section_headers": section_headers or [],

        # Populated after all chunks are created.
        "previous_chunk_id": None,
        "next_chunk_id": None,
    }


# ============================================================
# Main Stage 2 Function
# ============================================================

def recursive_structure_aware_chunking(
    ingestion_package: dict,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: Optional[list] = None,
    tokenizer_name: str = DEFAULT_TOKENIZER,
) -> dict:
    """
    Stage 2 — Recursive Structure-Aware Chunking.

    Input:
        Standardized output from Function 1.

    Output:
        Standardized chunk package.

    Guarantees:
        - Uses actual tiktoken token counts.
        - No final chunk exceeds chunk_size.
        - Overlap is created during chunk construction.
        - Overlap never crosses Markdown section boundaries.
        - Chunk relationships are explicitly recorded.
    """

    # --------------------------------------------------------
    # 1. Validate Stage 1 input
    # --------------------------------------------------------

    if not isinstance(
        ingestion_package,
        dict,
    ):

        return {
            "status": "failed",
            "error": (
                "ingestion_package must be "
                "a dictionary."
            ),
            "chunks": [],
        }

    if ingestion_package.get("status") != "processed":

        return {
            "status": "failed",
            "error": (
                "Stage 1 package must have "
                "status='processed'."
            ),
            "chunks": [],
        }

    content = ingestion_package.get(
        "content",
        {},
    )

    raw_text = content.get(
        "text",
        "",
    )

    if not isinstance(raw_text, str):

        return {
            "status": "failed",
            "error": "content.text must be a string.",
            "chunks": [],
        }

    raw_text = _normalize_text(raw_text)

    if not raw_text:

        return {
            "status": "chunked",
            "document": ingestion_package.get(
                "file",
                {},
            ),
            "document_metadata": ingestion_package.get(
                "metadata",
                {},
            ),
            "chunking": {
                "method": "recursive_structure_aware",
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "tokenizer": tokenizer_name,
            },
            "chunks": [],
            "chunk_count": 0,
        }

    # --------------------------------------------------------
    # 2. Validate configuration
    # --------------------------------------------------------

    if chunk_size <= 0:

        return {
            "status": "failed",
            "error": (
                "chunk_size must be greater "
                "than zero."
            ),
            "chunks": [],
        }

    if chunk_overlap < 0:

        return {
            "status": "failed",
            "error": (
                "chunk_overlap cannot be negative."
            ),
            "chunks": [],
        }

    if chunk_overlap >= chunk_size:

        return {
            "status": "failed",
            "error": (
                "chunk_overlap must be smaller "
                "than chunk_size."
            ),
            "chunks": [],
        }

    if separators is None:
        separators = DEFAULT_SEPARATORS

    # --------------------------------------------------------
    # 3. Build tokenizer once
    # --------------------------------------------------------

    try:

        token_counter = _build_token_counter(
            tokenizer_name
        )

    except Exception as error:

        return {
            "status": "failed",
            "error": (
                "Unable to initialize tokenizer: "
                f"{type(error).__name__}: {error}"
            ),
            "chunks": [],
        }

    # --------------------------------------------------------
    # 4. Detect Markdown structure
    # --------------------------------------------------------

    markdown_detected = bool(
        re.search(
            r"^#{1,6}\s+.+$",
            raw_text,
            flags=re.MULTILINE,
        )
    )

    if markdown_detected:

        sections = _split_markdown_sections(
            raw_text
        )

    else:

        sections = [{
            "text": raw_text,
            "headers": [],
        }]

    # --------------------------------------------------------
    # 5. Process each structural section independently
    # --------------------------------------------------------

    intermediate_chunks = []

    for section in sections:

        section_text = section["text"]
        section_headers = section["headers"]

        pieces = _recursive_split(
            text=section_text,
            chunk_size=chunk_size,
            separators=separators,
            token_counter=token_counter,
        )

        section_chunks = _construct_chunks_with_overlap(
            pieces=pieces,
            chunk_size=chunk_size,
            overlap_tokens=chunk_overlap,
            token_counter=token_counter,
        )

        for chunk_text in section_chunks:

            intermediate_chunks.append({
                "text": chunk_text,
                "section_headers": section_headers,
            })

    # --------------------------------------------------------
    # 6. Build final chunk objects
    # --------------------------------------------------------

    final_chunks = []

    for index, item in enumerate(
        intermediate_chunks,
        start=1,
    ):

        chunk_text = item["text"]

        metadata = _build_chunk_metadata(
            ingestion_package=ingestion_package,
            section_headers=item[
                "section_headers"
            ],
            chunk_index=index,
        )

        final_chunks.append({
            "chunk_id": metadata["chunk_id"],

            "text": chunk_text,

            "token_count": token_counter(
                chunk_text
            ),

            "metadata": metadata,
        })

    # --------------------------------------------------------
    # 7. Add chunk relationships
    # --------------------------------------------------------

    for index, chunk in enumerate(
        final_chunks
    ):

        previous_chunk = (
            final_chunks[index - 1]
            if index > 0
            else None
        )

        next_chunk = (
            final_chunks[index + 1]
            if index < len(final_chunks) - 1
            else None
        )

        chunk["metadata"][
            "previous_chunk_id"
        ] = (
            previous_chunk["chunk_id"]
            if previous_chunk
            else None
        )

        chunk["metadata"][
            "next_chunk_id"
        ] = (
            next_chunk["chunk_id"]
            if next_chunk
            else None
        )

    # --------------------------------------------------------
    # 8. Absolute final token-limit validation
    # --------------------------------------------------------

    for chunk in final_chunks:

        if chunk["token_count"] > chunk_size:

            return {
                "status": "failed",
                "error": (
                    "Internal chunking error: "
                    f"{chunk['chunk_id']} contains "
                    f"{chunk['token_count']} tokens, "
                    f"exceeding the configured limit "
                    f"of {chunk_size}."
                ),
                "chunks": [],
            }

    # --------------------------------------------------------
    # 9. Final standardized output
    # --------------------------------------------------------

    return {
        "status": "chunked",

        "document": ingestion_package.get(
            "file",
            {},
        ),

        "document_metadata": ingestion_package.get(
            "metadata",
            {},
        ),

        "chunking": {
            "method": "recursive_structure_aware",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "tokenizer": tokenizer_name,
            "separators": separators,
            "markdown_structure_detected": (
                markdown_detected
            ),
        },

        "chunks": final_chunks,

        "chunk_count": len(final_chunks),
    }