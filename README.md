# Defensive Custom RAG Pipeline

A production-grade, modular Retrieval-Augmented Generation (RAG) pipeline built entirely from scratch in Python with zero framework bloat (no LangChain, no LlamaIndex). Designed for low-latency local execution with strict input validation, defensive error boundary handling, and transactional database persistence.

## Architecture Highlights

The pipeline is split into six decoupled, independently testable stages:

1. **Stage 1: Ingestion & Routing** — Validates source files, inspects binary/text signatures, and handles encoding/parsing safely.
2. **Stage 2: Structure-Aware Chunking** — Token-bounded splitting using `tiktoken` with header graph tracking (`previous_chunk_id` and `next_chunk_id`).
3. **Stage 3: Embedding Batching** — Multi-batch vector generation with strict floating-point validation (`math.isfinite`) and NaN/Infinity defense.
4. **Stage 4: Dual Local Persistence** — Atomic storage splitting: dense vector arrays in **ChromaDB** and pristine JSON payloads in **SQLite**.
5. **Stage 5: Online Retrieval Engine** — Vector similarity search, score normalization, thresholding, and graph-expanded SQLite chunk hydration.
6. **Stage 6: Context Assembly & Generation** — Single-pass token budgeter with safety margins, source citation tagging, and metadata-safe Ollama response streaming.

## Installation & Setup

1. **Install Python Dependencies:**
   ```bash
   pip install -r requirements.txt
## Document synchronization

The pipeline keeps a local SQLite `rag_documents` manifest for active document versions.
Each source path gets a stable `document_id`, and each physical file is fingerprinted with SHA-256.

- `ingest` remains a **force-ingest** operation for one file.
- `ingest-dir` remains the original **force-ingest directory** command.
- `sync` is the hash-aware directory synchronization command.
- Same source path + same content hash → skipped.
- Same source path + changed content hash → reprocessed and re-embedded.
- Chunks no longer belonging to the current version are removed from the active SQLite/Chroma representation.
- The physical source file is never deleted by synchronization.

Stage 1 remains the authority for file routing and parsing. The synchronization logic lives in `RAGPipeline`; the CLI only exposes the existing workflow plus the new `sync` command.

### Example

```powershell
python main.py sync "data/documents"
python main.py ingest "data/documents/EXP 4.docx"
python main.py ingest-dir "data/documents"
python main.py ask "What is discussed in EXP 4?"
```

### Local Ollama models

The default generation model is `qwen3:4b`. The embedding model is configurable through `--embedding-model` or `RAG_EMBEDDING_MODEL` and must be the same embedding space used for document and query embeddings.

OCR support requires the Tesseract system executable. Scanned-PDF workflows may additionally require Poppler depending on the local Windows installation.
