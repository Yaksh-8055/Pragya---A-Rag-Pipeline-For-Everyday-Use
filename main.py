"""
Interactive entry point for the local RAG system.

Normal behavior:
    python main.py

Startup flow:
    1. Open the local RAG pipeline.
    2. Scan data/documents recursively.
    3. Hash every document.
    4. Embed/index only new or changed documents.
    5. Skip documents whose stored content hash is unchanged.
    6. Open an interactive question prompt.

The original explicit CLI commands are still available for debugging/manual use:
    python main.py ingest ...
    python main.py ingest-dir ...
    python main.py sync ...
    python main.py ask ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.pipeline import RAGPipeline

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "nomic-embed-text")
DEFAULT_DOCUMENT_ROOT = Path(os.getenv("RAG_DOCUMENT_ROOT", "data/documents"))
DEFAULT_STORAGE_ROOT = Path(os.getenv("RAG_STORAGE_ROOT", "data/storage"))
DEFAULT_LLM_MODEL = os.getenv("RAG_LLM_MODEL", "qwen2.5:7b")


def _ollama_embedding_client(
    texts: list[str],
    *,
    model: str,
    base_url: str,
) -> list[list[float]]:
    """Call Ollama /api/embed, with compatibility fallback to /api/embeddings."""
    if not isinstance(texts, list) or not texts:
        raise ValueError("texts must be a non-empty list.")
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("Every embedding input must be a string.")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Embedding model must be a non-empty string.")

    base = base_url.rstrip("/")
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    request = Request(
        f"{base}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=600) as response:
            data = json.loads(response.read().decode("utf-8"))
        embeddings = data.get("embeddings")
        if isinstance(embeddings, list) and len(embeddings) == len(texts):
            return embeddings
        raise RuntimeError("Ollama returned an invalid embedding count.")
    except HTTPError as exc:
        if exc.code != 404:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama embedding HTTP error {exc.code}: {body}"
            ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Could not connect to Ollama at {base}. Start Ollama first."
        ) from exc

    vectors: list[list[float]] = []
    for text in texts:
        legacy_payload = json.dumps(
            {"model": model, "prompt": text}
        ).encode("utf-8")
        legacy_request = Request(
            f"{base}/api/embeddings",
            data=legacy_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(legacy_request, timeout=600) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError) as exc:
            raise RuntimeError(
                f"Ollama legacy embedding request failed: {exc}"
            ) from exc

        vector = data.get("embedding")
        if not isinstance(vector, list):
            raise RuntimeError(
                "Ollama did not return an embedding vector."
            )
        vectors.append(vector)

    return vectors


def build_pipeline(args: argparse.Namespace) -> RAGPipeline:
    ollama_url = args.ollama_url.rstrip("/")

    def embedding_client(texts: list[str]) -> list[list[float]]:
        return _ollama_embedding_client(
            texts,
            model=args.embedding_model,
            base_url=ollama_url,
        )

    return RAGPipeline(
        embedding_client=embedding_client,
        storage_root=args.storage_root,
        ollama_model=args.llm_model,
        ollama_url=f"{ollama_url}/api/chat",
    )


def _print_json(value: object) -> None:
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def _iter_document_files(
    root: Path,
    *,
    recursive: bool,
) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        p
        for p in iterator
        if p.is_file()
        and not p.name.startswith("~$")
        and not p.name.startswith(".")
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    with build_pipeline(args) as pipeline:
        result = pipeline.ingest(
            args.path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_batch_size=args.embedding_batch_size,
        )

    _print_json(result)
    return 0 if result.get("status") == "success" else 1


def cmd_ingest_dir(args: argparse.Namespace) -> int:
    with build_pipeline(args) as pipeline:
        results = pipeline.ingest_directory(
            args.path,
            recursive=not args.no_recursive,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_batch_size=args.embedding_batch_size,
        )

    summary = {
        "total_files": len(results),
        "successful": sum(
            r.get("status") == "success" for r in results
        ),
        "failed": sum(
            r.get("status") != "success" for r in results
        ),
        "results": results,
    }
    _print_json(summary)
    return 0 if summary["failed"] == 0 else 1


def cmd_sync(args: argparse.Namespace) -> int:
    root = args.path
    root.mkdir(parents=True, exist_ok=True)
    files = _iter_document_files(
        root,
        recursive=not args.no_recursive,
    )

    if not files:
        print(f"No files found under: {root.resolve()}")
        return 0

    with build_pipeline(args) as pipeline:
        results = [
            pipeline.sync_file(
                path,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                embedding_batch_size=args.embedding_batch_size,
            )
            for path in files
        ]

    summary = {
        "total_files": len(results),
        "ingested": sum(
            r.get("status") == "success" for r in results
        ),
        "skipped": sum(
            r.get("status") == "skipped" for r in results
        ),
        "failed": sum(
            r.get("status") == "failed" for r in results
        ),
        "results": results,
    }
    _print_json(summary)
    return 0 if summary["failed"] == 0 else 1


def cmd_ask(args: argparse.Namespace) -> int:
    with build_pipeline(args) as pipeline:
        result = pipeline.ask(
            args.query,
            top_k=args.top_k,
            similarity_threshold=args.similarity_threshold,
            max_prompt_tokens=args.max_prompt_tokens,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            stream=args.stream,
        )

        if args.stream:
            for event in result:
                if event.get("type") == "token":
                    print(
                        event.get("content", ""),
                        end="",
                        flush=True,
                    )
                elif event.get("type") == "metadata":
                    print()
                    _print_json(event)
            return 0

    _print_json(result)
    return 0 if result.get("status") == "success" else 1


def _make_interactive_args(
    parser: argparse.ArgumentParser,
    argv: list[str],
) -> argparse.Namespace:
    """Parse only global options for the normal interactive mode."""
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        parser.error(
            "interactive mode only accepts global options before a command; "
            f"unknown arguments: {' '.join(unknown)}"
        )
    return args


def _startup_sync(args: argparse.Namespace) -> tuple[dict, RAGPipeline]:
    """Open the pipeline, synchronize all documents, and keep it open for queries."""
    root = DEFAULT_DOCUMENT_ROOT
    root.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline(args)

    try:
        files = _iter_document_files(
            root,
            recursive=True,
        )

        if not files:
            print(f"No documents found under: {root.resolve()}")
            return {
                "total_files": 0,
                "ingested": 0,
                "skipped": 0,
                "failed": 0,
                "results": [],
            }, pipeline

        print(f"Checking {len(files)} document(s) under: {root.resolve()}")
        print("Hash-checking and indexing new/changed documents...\n")

        results: list[dict] = []

        for path in files:
            print(f"  Checking: {path.name}", end=" ... ", flush=True)

            result = pipeline.sync_file(
                path,
                chunk_size=700,
                chunk_overlap=70,
                embedding_batch_size=32,
            )
            results.append(result)

            status = result.get("status")
            if status == "skipped":
                print("SKIPPED (unchanged)")
            elif status == "success":
                print("EMBEDDED / UPDATED")
            else:
                print("FAILED")
                print(
                    f"    {result.get('error', 'Unknown error')}"
                )

        summary = {
            "total_files": len(results),
            "ingested": sum(
                r.get("status") == "success" for r in results
            ),
            "skipped": sum(
                r.get("status") == "skipped" for r in results
            ),
            "failed": sum(
                r.get("status") == "failed" for r in results
            ),
            "results": results,
        }

        return summary, pipeline

    except Exception:
        pipeline.close()
        raise


def _interactive_chat(args: argparse.Namespace) -> int:
    """Normal application mode: sync first, then continuously accept questions."""
    summary, pipeline = _startup_sync(args)

    try:
        print("\n---------------------------------------------")
        print("RAG startup complete")
        print(
            f"Documents: {summary['total_files']} | "
            f"Embedded/updated: {summary['ingested']} | "
            f"Skipped: {summary['skipped']} | "
            f"Failed: {summary['failed']}"
        )
        print("---------------------------------------------")
        print("Ask questions about your documents.")
        print("Type 'exit' or 'quit' to close.\n")

        while True:
            try:
                query = input("You: ").strip()
            except EOFError:
                print()
                break

            if not query:
                continue

            if query.lower() in {"exit", "quit"}:
                break

            result = pipeline.ask(
                query,
                top_k=None,
                similarity_threshold=None,
                max_prompt_tokens=3500,
                max_output_tokens=600,
                temperature=0.2,
                stream=True,
            )

            print("\nRAG: ", end="", flush=True)

            if isinstance(result, dict):
                if result.get("status") == "failed":
                    print()
                    print(
                        f"Error: {result.get('error', 'Unknown error')}"
                    )
                    print()
                    continue

                print(result.get("answer", ""))
                print()
                continue

            metadata = None
            print("", end="")

            try:
                for event in result:
                    event_type = event.get("type")

                    if event_type == "token":
                        print(
                            event.get("content", ""),
                            end="",
                            flush=True,
                        )
                    elif event_type == "metadata":
                        metadata = event

            except Exception as exc:
                print()
                print(
                    f"Error while generating answer: "
                    f"{type(exc).__name__}: {exc}"
                )
                print()
                continue

            print("\n")

            if metadata and metadata.get("status") == "failed":
                print(
                    f"Error: {metadata.get('error', 'Unknown error')}"
                )
                print()

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        pipeline.close()

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Local file-to-RAG pipeline. "
            "Running without a command automatically synchronizes "
            "data/documents and opens the interactive question prompt."
        )
    )

    parser.add_argument(
        "--storage-root",
        type=Path,
        default=DEFAULT_STORAGE_ROOT,
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
    )

    sub = parser.add_subparsers(dest="command", required=False)

    ingest = sub.add_parser(
        "ingest",
        help="Force-ingest one file.",
    )
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--chunk-size", type=int, default=700)
    ingest.add_argument("--chunk-overlap", type=int, default=70)
    ingest.add_argument("--embedding-batch-size", type=int, default=32)
    ingest.set_defaults(func=cmd_ingest)

    ingest_dir = sub.add_parser(
        "ingest-dir",
        help="Force-ingest all files in a directory.",
    )
    ingest_dir.add_argument("path", type=Path)
    ingest_dir.add_argument("--no-recursive", action="store_true")
    ingest_dir.add_argument("--chunk-size", type=int, default=700)
    ingest_dir.add_argument("--chunk-overlap", type=int, default=70)
    ingest_dir.add_argument("--embedding-batch-size", type=int, default=32)
    ingest_dir.set_defaults(func=cmd_ingest_dir)

    sync = sub.add_parser(
        "sync",
        help="Synchronize a directory using content hashes.",
    )
    sync.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=DEFAULT_DOCUMENT_ROOT,
    )
    sync.add_argument("--no-recursive", action="store_true")
    sync.add_argument("--chunk-size", type=int, default=700)
    sync.add_argument("--chunk-overlap", type=int, default=70)
    sync.add_argument("--embedding-batch-size", type=int, default=32)
    sync.set_defaults(func=cmd_sync)

    ask = sub.add_parser(
        "ask",
        help="Ask the local RAG system.",
    )
    ask.add_argument("query")
    ask.add_argument("--top-k", type=int, default=None)
    ask.add_argument("--similarity-threshold", type=float, default=None)
    ask.add_argument("--max-prompt-tokens", type=int, default=3500)
    ask.add_argument("--max-output-tokens", type=int, default=600)
    ask.add_argument("--temperature", type=float, default=0.2)
    ask.add_argument("--stream", action="store_true")
    ask.set_defaults(func=cmd_ask)

    return parser


def main() -> int:
    parser = build_parser()

    # No command = normal application mode.
    # This is intentionally the default UX for `python main.py`.
    if len(sys.argv) == 1:
        args = argparse.Namespace(
            storage_root=DEFAULT_STORAGE_ROOT,
            ollama_url=DEFAULT_OLLAMA_URL,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            llm_model=DEFAULT_LLM_MODEL,
        )
        return _interactive_chat(args)

    args = parser.parse_args()

    # Global flags can also be used without a command:
    # `python main.py --llm-model qwen3:4b`
    if args.command is None:
        return _interactive_chat(args)

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
