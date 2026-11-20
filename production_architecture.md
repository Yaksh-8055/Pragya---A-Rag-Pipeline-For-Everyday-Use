# RAG Pipeline Ingestion Framework

## Stage 1: File Routing
Processes text, layout PDFs, and structured inputs.

## Stage 2: Recursive Chunking
Enforces exact token boundaries using tiktoken while tracking headers.

## Stage 3: Embedding Batching
Generates dense floating-point vector representations.

## Stage 4: Dual Persistence
Indexes vectors in ChromaDB and persists full document payloads in SQLite.