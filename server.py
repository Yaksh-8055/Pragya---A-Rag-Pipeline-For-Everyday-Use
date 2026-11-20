"""
Local HTTP service around the RAG pipeline  (Python standard library only).

Put this file in the RAG project root, next to its main.py, and run it with the
RAG project's own Python environment:

    python server.py                 # sync documents in the background, serve on 127.0.0.1:8765
    python server.py --no-sync       # serve only, don't scan data/documents
    python server.py --port 9000

Endpoints (JSON in, JSON out):
    GET  /health        service + indexing status (never blocks on the pipeline)
    POST /query         {"question": "...", "top_k": 8, "voice": true}
                        -> {"answer", "spoken", "found", "sources", "timings"}
    POST /find          {"query": "...", "limit": 3}
                        -> {"files": [{"file_name", "file_path", "score"}]}
    POST /sync          start a background hash-aware sync of data/documents
    GET  /sync/status   progress of that sync

Design notes
  * ONE worker thread owns the pipeline. SQLite connections can't cross threads,
    and Chroma/SQLite want a single writer. Queries and per-file sync jobs share
    that thread's queue, so a question waits at most one file's ingestion time.
  * Binds to 127.0.0.1 only, checks the Host header and requires a JSON
    content type, so a web page open in your browser cannot talk to it.
  * Your pipeline code is not modified: retrieval and generation are your own
    pipeline.retrieve() and generate_answer(), called with a voice-style question.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_PORT = 8765
MAX_BODY_BYTES = 64 * 1024
QUERY_TIMEOUT_S = 300

# Appended to the question ONLY for generation; retrieval uses the raw question.
VOICE_SUFFIX = (
    "\n\n(Voice answer: at most three short sentences in plain spoken English. "
    "No markdown, lists, headings or file names. If the context does not contain "
    "the answer, say exactly: I do not have enough information to answer this.)"
)
_NO_INFO = re.compile(r"^\W*i do not have enough information", re.I)

# Never offered to the "open this file" feature.
_UNSAFE_EXT = {
    ".exe", ".bat", ".cmd", ".ps1", ".vbs", ".vbe", ".js", ".jse", ".wsf", ".msi",
    ".scr", ".hta", ".jar", ".com", ".lnk", ".reg", ".dll", ".sys",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def to_spoken(text: str, max_sentences: int = 4, max_chars: int = 500) -> str:
    """Markdown-free, short version of an answer, safe to read aloud."""
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", "", text, flags=re.M)
    text = re.sub(r"[*`~]+", "", text)
    text = re.sub(r"(?<!\w)_+|_+(?!\w)", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s]
    kept = sentences[:max_sentences]
    while len(kept) > 1 and len(" ".join(kept)) > max_chars:
        kept.pop()
    return " ".join(kept)


def _source_entry(result: dict) -> dict:
    md = result.get("metadata") or {}
    headers = md.get("section_headers") or []
    if isinstance(headers, str):
        headers = [headers]
    return {
        "file_name": md.get("file_name") or "unknown",
        "file_path": md.get("file_path"),
        "section": " > ".join(str(h) for h in headers if str(h).strip()),
        "score": round(float(result.get("similarity_score") or 0.0), 4),
    }


def _dedupe_by_file(entries: list) -> list:
    best = {}
    for e in entries:
        key = e["file_path"] or e["file_name"]
        if key not in best or e["score"] > best[key]["score"]:
            best[key] = e
    return sorted(best.values(), key=lambda e: e["score"], reverse=True)


# ---------------------------------------------------------------------------
# service (owns the pipeline; every pipeline call runs on ONE worker thread)
# ---------------------------------------------------------------------------
class RagService:
    def __init__(self, cli_args_factory=None):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag")
        self._pipeline = None
        self._init_error = None
        self._cli_args_factory = cli_args_factory
        self.counts = {"documents": None, "chunks": None}
        self.sync_state = {
            "running": False, "total": 0, "done": 0, "ingested": 0,
            "skipped": 0, "failed": 0, "last_error": None, "finished_at": None,
        }
        self._sync_lock = threading.Lock()

    # ---- worker-thread only ------------------------------------------------
    def _pipe(self):
        if self._pipeline is None:
            import main as rag_main  # the RAG project's main.py
            args = (self._cli_args_factory() if self._cli_args_factory else
                    argparse.Namespace(
                        storage_root=rag_main.DEFAULT_STORAGE_ROOT,
                        ollama_url=rag_main.DEFAULT_OLLAMA_URL,
                        embedding_model=rag_main.DEFAULT_EMBEDDING_MODEL,
                        llm_model=rag_main.DEFAULT_LLM_MODEL,
                    ))
            self._pipeline = rag_main.build_pipeline(args)
            self._refresh_counts()
        return self._pipeline

    def _refresh_counts(self):
        try:
            p = self._pipeline
            self.counts = {
                "documents": p.sqlite_connection.execute(
                    "SELECT COUNT(*) FROM rag_documents").fetchone()[0],
                "chunks": int(p.chroma_collection.count()),
            }
        except Exception:
            pass

    def _submit(self, fn, timeout=QUERY_TIMEOUT_S):
        def job():
            return fn(self._pipe())
        return self._pool.submit(job).result(timeout=timeout)

    # ---- public API --------------------------------------------------------
    def warm_up(self):
        try:
            self._submit(lambda p: None, timeout=120)
        except Exception as e:
            self._init_error = f"{type(e).__name__}: {e}"
            print(f"[rag-server] pipeline failed to start: {self._init_error}")

    def health(self) -> dict:
        return {
            "ok": self._init_error is None,
            "pipeline_ready": self._pipeline is not None,
            "error": self._init_error,
            "counts": self.counts,
            "sync": dict(self.sync_state),
        }

    def query(self, question, top_k=8, threshold=None, max_output_tokens=250, voice=True):
        def run(pipe):
            from src.stage_6_generation.generator import generate_answer

            t0 = time.perf_counter()
            retrieval = pipe.retrieve(question, top_k=top_k, similarity_threshold=threshold)
            t1 = time.perf_counter()
            if retrieval.get("status") == "failed":
                return {"ok": False, "error": retrieval.get("error", "Retrieval failed.")}

            gen = generate_answer(
                question + VOICE_SUFFIX if voice else question,
                retrieval,
                llm_client=pipe._ollama_client,
                max_prompt_tokens=3500,
                max_output_tokens=max_output_tokens,
                temperature=0.2,
                stream=False,
            )
            t2 = time.perf_counter()
            if gen.get("status") != "success":
                return {"ok": False, "error": gen.get("error", "Generation failed.")}

            by_chunk = {r["chunk_id"]: r for r in retrieval.get("results", [])}
            sources = _dedupe_by_file([
                _source_entry(by_chunk[s["chunk_id"]])
                for s in gen.get("sources", []) if s.get("chunk_id") in by_chunk
            ])
            answer = gen.get("answer", "")
            return {
                "ok": True,
                "answer": answer,
                "spoken": to_spoken(answer) if voice else answer,
                "found": bool(answer.strip()) and not _NO_INFO.match(answer),
                "sources": sources,
                "timings": {"retrieval_s": round(t1 - t0, 2), "generation_s": round(t2 - t1, 2)},
            }
        return self._submit(run)

    def find(self, query, limit=3, threshold=None):
        def run(pipe):
            retrieval = pipe.retrieve(query, top_k=max(limit * 4, 8),
                                      similarity_threshold=threshold, expand_context=False)
            if retrieval.get("status") == "failed":
                return {"ok": False, "error": retrieval.get("error", "Retrieval failed.")}
            files = []
            for e in _dedupe_by_file([_source_entry(r) for r in retrieval.get("results", [])]):
                p = e["file_path"]
                if not p or Path(p).suffix.lower() in _UNSAFE_EXT or not Path(p).is_file():
                    continue
                files.append({"file_name": e["file_name"], "file_path": p, "score": e["score"]})
            return {"ok": True, "files": files[:limit]}
        return self._submit(run)

    # ---- background sync (per-file jobs, so queries can interleave) --------
    def start_sync(self) -> bool:
        with self._sync_lock:
            if self.sync_state["running"]:
                return False
            self.sync_state.update(running=True, total=0, done=0, ingested=0,
                                   skipped=0, failed=0, last_error=None)
        threading.Thread(target=self._sync_loop, daemon=True, name="rag-sync").start()
        return True

    def _sync_loop(self):
        try:
            import main as rag_main
            root = rag_main.DEFAULT_DOCUMENT_ROOT
            root.mkdir(parents=True, exist_ok=True)
            files = rag_main._iter_document_files(root, recursive=True)
            self.sync_state["total"] = len(files)
            for path in files:
                try:
                    res = self._submit(
                        lambda p, path=path: p.sync_file(
                            path, chunk_size=700, chunk_overlap=70, embedding_batch_size=32),
                        timeout=3600)
                except Exception as e:
                    res = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
                status = res.get("status")
                key = {"success": "ingested", "skipped": "skipped"}.get(status, "failed")
                self.sync_state[key] += 1
                self.sync_state["done"] += 1
                if key == "failed":
                    self.sync_state["last_error"] = f"{path.name}: {res.get('error', 'failed')}"
                try:
                    self._submit(lambda p: self._refresh_counts(), timeout=60)
                except Exception:
                    pass
        except Exception as e:
            self.sync_state["last_error"] = f"{type(e).__name__}: {e}"
        finally:
            self.sync_state["running"] = False
            self.sync_state["finished_at"] = time.time()

    def close(self):
        try:
            if self._pipeline is not None:
                self._pool.submit(self._pipeline.close).result(timeout=30)
        except Exception:
            pass
        self._pool.shutdown(wait=False)


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    service: RagService = None  # set by make_server
    server_version = "SarvagyaRAG/1.0"

    def log_message(self, fmt, *args):
        print(f"[rag-server] {self.address_string()} {fmt % args}")

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].lower()
        return host in {"127.0.0.1", "localhost"}

    def _read_json(self):
        if "application/json" not in (self.headers.get("Content-Type") or ""):
            self._send(415, {"ok": False, "error": "Content-Type must be application/json."})
            return None
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY_BYTES:
            self._send(413, {"ok": False, "error": "Request body too large."})
            return None
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"ok": False, "error": "Invalid JSON."})
            return None
        if not isinstance(data, dict):
            self._send(400, {"ok": False, "error": "JSON body must be an object."})
            return None
        return data

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, {"ok": False, "error": "Forbidden host."})
        if self.path == "/health":
            return self._send(200, self.service.health())
        if self.path == "/sync/status":
            return self._send(200, dict(self.service.sync_state))
        self._send(404, {"ok": False, "error": "Not found."})

    def do_POST(self):
        if not self._host_ok():
            return self._send(403, {"ok": False, "error": "Forbidden host."})
        data = self._read_json()
        if data is None:
            return
        try:
            if self.path == "/query":
                q = data.get("question")
                if not isinstance(q, str) or not q.strip() or len(q) > 2000:
                    return self._send(400, {"ok": False, "error": "'question' must be a non-empty string."})
                result = self.service.query(
                    q.strip(),
                    top_k=_int(data.get("top_k"), 8, 1, 50),
                    threshold=_float_or_none(data.get("similarity_threshold")),
                    max_output_tokens=_int(data.get("max_output_tokens"), 250, 32, 600),
                    voice=bool(data.get("voice", True)),
                )
            elif self.path == "/find":
                q = data.get("query")
                if not isinstance(q, str) or not q.strip() or len(q) > 500:
                    return self._send(400, {"ok": False, "error": "'query' must be a non-empty string."})
                result = self.service.find(
                    q.strip(), limit=_int(data.get("limit"), 3, 1, 10),
                    threshold=_float_or_none(data.get("similarity_threshold")))
            elif self.path == "/sync":
                result = {"ok": True, "started": self.service.start_sync()}
            else:
                return self._send(404, {"ok": False, "error": "Not found."})
            self._send(200 if result.get("ok") else 502, result)
        except TimeoutError:
            self._send(504, {"ok": False, "error": "Timed out waiting for the pipeline."})
        except Exception as e:
            if self.service._init_error:
                return self._send(503, {"ok": False, "error": self.service._init_error})
            self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})


def _int(v, default, lo, hi):
    if isinstance(v, bool) or not isinstance(v, int):
        return default
    return max(lo, min(hi, v))


def _float_or_none(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def make_server(service: RagService, host="127.0.0.1", port=DEFAULT_PORT) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"service": service})
    return ThreadingHTTPServer((host, port), handler)


def main() -> int:
    ap = argparse.ArgumentParser(description="Local HTTP service for the RAG pipeline.")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-sync", action="store_true", help="don't scan data/documents at startup")
    args = ap.parse_args()

    service = RagService()
    print("[rag-server] starting pipeline (Ollama must be running)...")
    service.warm_up()
    if not args.no_sync and service._init_error is None:
        service.start_sync()
        print("[rag-server] background document sync started.")

    server = make_server(service, "127.0.0.1", args.port)
    print(f"[rag-server] listening on http://127.0.0.1:{args.port}   (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[rag-server] shutting down...")
    finally:
        server.server_close()
        service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())