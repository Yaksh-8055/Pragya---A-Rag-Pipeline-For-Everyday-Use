# Deep Dive into RAG Architecture

## Stage 1: Ingestion
The ingestion engine acts as the gatekeeper. It routes documents based on extension, 
handles character encoding, and strips out unsupported media formats like audio or video.

## Stage 2: Recursive Chunking
Text chunking breaks documents into token-bounded pieces using tiktoken.
It respects header hierarchies (# and ##) and maintains previous and next chunk ID linkages.

## Stage 3: Embedding Generation
Embedding generation converts clean text chunks into high-dimensional vectors.
It enforces strict dimension consistency, guards against NaN/Infinity, and batches requests safely.
