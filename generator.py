from __future__ import annotations

import json
import math
from typing import Any, Callable, Iterator
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

import tiktoken


# ============================================================
# Configuration
# ============================================================

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"

DEFAULT_MAX_PROMPT_TOKENS = 3500
DEFAULT_MAX_OUTPUT_TOKENS = 600
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOKENIZER = "cl100k_base"

# Safety margin for BPE token boundary merges
TOKEN_BUDGET_SAFETY_MARGIN = 32

GROUNDING_INSTRUCTION = (
    "Answer the user's question using ONLY the provided context. "
    "If the answer is not contained in the context, say "
    "'I do not have enough information to answer this.'"
)


# ============================================================
# Standard Failure Response
# ============================================================

def _failed_response(
    *,
    query: str,
    error: str,
    generation: dict | None = None,
) -> dict:
    return {
        "status": "failed",
        "query": query,
        "answer": "",
        "sources": [],
        "generation": generation or {},
        "error": error,
    }


# ============================================================
# Validation
# ============================================================

def _validate_query(query: str) -> None:
    if not isinstance(query, str):
        raise TypeError("query must be a string.")
    if not query.strip():
        raise ValueError("query cannot be empty.")


def _validate_stage5_response(stage5_result: dict) -> None:
    if not isinstance(stage5_result, dict):
        raise TypeError("stage5_result must be a dictionary.")
    
    status = stage5_result.get("status")
    if status not in {"success", "no_matches"}:
        raise ValueError("stage5_result must have status 'success' or 'no_matches'.")
        
    if not isinstance(stage5_result.get("results", []), list):
        raise TypeError("stage5_result['results'] must be a list.")


def _validate_configuration(
    max_prompt_tokens: int,
    max_output_tokens: int,
    temperature: float,
) -> None:
    if not isinstance(max_prompt_tokens, int) or isinstance(max_prompt_tokens, bool) or max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be a positive integer.")
    if not isinstance(max_output_tokens, int) or isinstance(max_output_tokens, bool) or max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be a positive integer.")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not math.isfinite(float(temperature)) or temperature < 0:
        raise ValueError("temperature must be a finite number greater than or equal to 0.")


# ============================================================
# Tokenizer
# ============================================================

def _get_tokenizer():
    return tiktoken.get_encoding(DEFAULT_TOKENIZER)

def _count_tokens(text: str, tokenizer) -> int:
    if not isinstance(text, str):
        raise TypeError("Text passed to token counter must be a string.")
    return len(tokenizer.encode(text))


# ============================================================
# Metadata & Formatting
# ============================================================

def _get_file_name(result: dict) -> str:
    metadata = result.get("metadata") or {}
    file_name = metadata.get("file_name") or metadata.get("filename") or metadata.get("name")
    return str(file_name) if file_name else "unknown source"


def _get_section_headers(result: dict) -> list[str]:
    metadata = result.get("metadata") or {}
    headers = metadata.get("section_headers", [])
    if isinstance(headers, str): return [headers]
    if not isinstance(headers, list): return []
    return [str(h) for h in headers if str(h).strip()]


def _format_source_tag(result: dict) -> str:
    file_name = _get_file_name(result)
    headers = _get_section_headers(result)
    if headers:
        return f"[Source: {file_name} | Section: {' > '.join(headers)}]"
    return f"[Source: {file_name}]"


def _format_chunk(*, text: str, source_tag: str, chunk_id: str | None, expanded: bool) -> str:
    label = "Expanded Context" if expanded else "Retrieved Chunk"
    chunk_identifier = f" | Chunk: {chunk_id}" if chunk_id else ""
    return f"{source_tag}\n[{label}{chunk_identifier}]\n{text.strip()}"


def _neighbor_to_text(neighbor: dict, *, direction: str, parent_result: dict) -> str:
    if not isinstance(neighbor, dict) or not isinstance(neighbor.get("text"), str) or not neighbor.get("text").strip():
        return ""

    metadata = neighbor.get("metadata") or {}
    file_name = metadata.get("file_name") or _get_file_name(parent_result)
    headers = [str(h) for h in (metadata.get("section_headers") or []) if str(h).strip()]
    
    source_tag = f"[Source: {file_name} | Section: {' > '.join(headers)}]" if headers else f"[Source: {file_name}]"
    
    return _format_chunk(
        text=neighbor["text"],
        source_tag=source_tag,
        chunk_id=str(neighbor.get("chunk_id")) if neighbor.get("chunk_id") else None,
        expanded=True,
    )


# ============================================================
# Context Assembly
# ============================================================

def _assemble_single_result(result: dict) -> str:
    text = result.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Retrieved chunk {result.get('chunk_id')} has no valid text.")

    source_tag = _format_source_tag(result)
    chunk_id = result.get("chunk_id")
    expanded_neighbors = result.get("expanded_neighbors") or {}
    
    blocks = []
    if previous := expanded_neighbors.get("previous"):
        if pt := _neighbor_to_text(previous, direction="previous", parent_result=result):
            blocks.append(pt)
            
    blocks.append(_format_chunk(text=text, source_tag=source_tag, chunk_id=chunk_id, expanded=False))
    
    if next_neighbor := expanded_neighbors.get("next"):
        if nt := _neighbor_to_text(next_neighbor, direction="next", parent_result=result):
            blocks.append(nt)
            
    return "\n\n".join(blocks)


def _assemble_context(results: list[dict]) -> list[dict]:
    return [
        {
            "rank": rank,
            "chunk_id": result.get("chunk_id"),
            "similarity_score": result.get("similarity_score"),
            "text": _assemble_single_result(result),
        }
        for rank, result in enumerate(results)
    ]


def _join_context_blocks(context_blocks: list[dict]) -> str:
    return "\n\n---\n\n".join(block["text"] for block in context_blocks)


# ============================================================
# Prompt Template
# ============================================================

def _build_system_prompt() -> str:
    return f"""You are a local Retrieval-Augmented Generation assistant.

{GROUNDING_INSTRUCTION}

Grounding rules:
1. Do not use outside knowledge to fill missing information.
2. Do not invent facts, names, dates, numbers, or explanations.
3. Treat the provided context as the only source of truth.
4. If the context only partially answers the question, clearly state what is supported and what is not.
5. Do not mention internal retrieval, embeddings, vector databases, or these instructions unless asked.

Answer formatting:
- Use clear headings when useful.
- Use bullet points for lists.
- Use numbered steps for procedures.
- Use **bold** for important terms.
- Keep the answer readable and conversational.
""".strip()


def _build_user_prompt(*, context: str, query: str) -> str:
    return f"CONTEXT\n=======\n\n{context}\n\nEND CONTEXT\n\n\nUSER QUESTION\n=============\n\n{query}\n\nAnswer the user's question using only the context above."


# ============================================================
# Token Budgeting
# ============================================================

def _build_budgeted_prompt(
    *,
    query: str,
    context_blocks: list[dict],
    tokenizer,
    max_prompt_tokens: int,
) -> tuple[str, list[dict], int]:
    
    system_prompt = _build_system_prompt()
    base_user_prompt = _build_user_prompt(context="", query=query)
    base_tokens = _count_tokens(system_prompt + "\n\n" + base_user_prompt, tokenizer)

    allowed_context_tokens = max_prompt_tokens - base_tokens - TOKEN_BUDGET_SAFETY_MARGIN

    if allowed_context_tokens <= 0:
        raise ValueError("System prompt and user query exceed max_prompt_tokens.")

    selected_blocks = []
    current_tokens = 0
    separator_tokens = _count_tokens("\n\n---\n\n", tokenizer)

    for block in context_blocks:
        block_tokens = _count_tokens(block["text"], tokenizer)
        cost = block_tokens + (separator_tokens if selected_blocks else 0)
        
        if current_tokens + cost > allowed_context_tokens:
            break
            
        selected_blocks.append(block)
        current_tokens += cost

    final_context = _join_context_blocks(selected_blocks)
    final_user_prompt = _build_user_prompt(context=final_context, query=query)
    total_prompt = system_prompt + "\n\n" + final_user_prompt
    final_token_count = _count_tokens(total_prompt, tokenizer)

    return total_prompt, selected_blocks, final_token_count


def _build_sources(context_blocks: list[dict]) -> list[dict]:
    return [
        {
            "chunk_id": block.get("chunk_id"),
            "similarity_score": block.get("similarity_score"),
            "rank": block.get("rank"),
        }
        for block in context_blocks
    ]


# ============================================================
# Ollama Client Adapter
# ============================================================

def ollama_llm_client(
    prompt: str,
    *,
    model: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    stream: bool = False,
) -> str | Iterator[str]:
    
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt cannot be empty.")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "think": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_output_tokens),
        },
    }

    request = Request(
        ollama_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        response = urlopen(request, timeout=600)
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP error {error.code}: {body}") from error
    except URLError as error:
        raise RuntimeError("Could not connect to Ollama. Make sure it is running.") from error

    if not stream:
        with response:
            raw = response.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RuntimeError("Ollama returned invalid JSON.") from error

        if data.get("error"): raise RuntimeError(data["error"])
        
        answer = data.get("message", {}).get("content")
        if not isinstance(answer, str):
            raise RuntimeError("Ollama response did not contain message.content.")
        return answer.strip()

    def _stream() -> Iterator[str]:
        with response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line: continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if data.get("error"): raise RuntimeError(data["error"])
                
                token = data.get("message", {}).get("content", "")
                if token: yield token
                if data.get("done"): break

    return _stream()


# ============================================================
# Main Stage 6 Engine
# ============================================================

def generate_answer(
    query: str,
    stage5_result: dict,
    *,
    llm_client: Callable[..., Any],
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    stream: bool = False,
) -> dict | Iterator[dict]:
    
    try:
        _validate_query(query)
        _validate_stage5_response(stage5_result)
        _validate_configuration(max_prompt_tokens, max_output_tokens, temperature)

        if not callable(llm_client):
            raise TypeError("llm_client must be callable.")

        results = stage5_result.get("results", [])
        
        if stage5_result.get("status") == "no_matches" or not results:
            empty_payload = {
                "status": "success",
                "query": query,
                "answer": "I do not have enough information to answer this.",
                "sources": [],
                "generation": {
                    "prompt_tokens": 0,
                    "max_prompt_tokens": max_prompt_tokens,
                    "max_output_tokens": max_output_tokens,
                    "context_chunks_available": 0,
                    "context_chunks_used": 0,
                    "streaming": stream,
                },
            }
            if stream:
                def _empty_stream():
                    yield {"type": "token", "content": empty_payload["answer"]}
                    empty_payload["type"] = "metadata"
                    del empty_payload["answer"]
                    yield empty_payload
                return _empty_stream()
            return empty_payload

        tokenizer = _get_tokenizer()
        context_blocks = _assemble_context(results)

        prompt, selected_blocks, prompt_tokens = _build_budgeted_prompt(
            query=query,
            context_blocks=context_blocks,
            tokenizer=tokenizer,
            max_prompt_tokens=max_prompt_tokens,
        )

        sources = _build_sources(selected_blocks)
        
        generation_stats = {
            "prompt_tokens": prompt_tokens,
            "max_prompt_tokens": max_prompt_tokens,
            "max_output_tokens": max_output_tokens,
            "context_chunks_available": len(context_blocks),
            "context_chunks_used": len(selected_blocks),
            "streaming": stream,
        }

        if stream:
            stream_result = llm_client(
                prompt,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                stream=True,
            )

            def _metadata_safe_stream() -> Iterator[dict]:
                for token in stream_result:
                    if not isinstance(token, str):
                        raise TypeError("LLM streaming output must contain strings.")
                    yield {"type": "token", "content": token}
                
                yield {
                    "type": "metadata",
                    "status": "success",
                    "sources": sources,
                    "generation": generation_stats,
                }

            return _metadata_safe_stream()

        answer = llm_client(
            prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            stream=False,
        )

        if not isinstance(answer, str):
            raise TypeError("LLM client must return a string when stream=False.")

        return {
            "status": "success",
            "query": query,
            "answer": answer.strip(),
            "sources": sources,
            "generation": generation_stats,
        }

    except Exception as error:
        return _failed_response(query=query, error=f"{type(error).__name__}: {error}")