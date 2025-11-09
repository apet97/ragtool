# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Clockify RAG CLI** – A local, offline retrieval-augmented generation (RAG) system that answers questions about Clockify's documentation using a local Ollama instance.

- **Two implementations**: v1.0 (simple, educational) and v2.0 (production-ready, recommended)
- **Fully offline**: No external APIs; uses local Ollama at `http://127.0.0.1:11434` (configurable)
- **Knowledge source**: 6.9 MB Clockify markdown documentation (~150 pages)
- **User interface**: CLI with command-line interface (v1.0) or interactive REPL (v2.0)

## Architecture & Components

### High-Level Pipeline

```
Knowledge Base (knowledge_full.md)
    ↓
Chunking (split by ## headings, max 1600 chars)
    ↓
Embedding (nomic-embed-text model via Ollama)
    ↓
Storage (vecs.npy vectors + meta.jsonl metadata + BM25 index)
    ↓
Query (user question)
    ↓
Retrieval (v1.0: cosine similarity; v2.0: hybrid BM25+dense+MMR)
    ↓
LLM (qwen2.5:32b via Ollama to generate answer)
    ↓
Response (answer with citations or "I don't know based on the MD.")
```

### Core Files

| File | Version | Lines | Purpose |
|------|---------|-------|---------|
| `clockify_rag.py` | v1.0 | ~350 | Simple three-step CLI: chunk → embed → ask |
| `clockify_support_cli.py` | v2.0 | ~1400+ | Production-grade with hybrid retrieval, REPL, debug mode |
| `knowledge_full.md` | N/A | ~7.2 MB | Merged Clockify docs (input to chunking) |
| `chunks.jsonl` | Generated | Varies | JSONL format: `{"id": int, "text": str}` per line |
| `vecs.npy` | Generated | Binary | NumPy array [num_chunks, 768] (normalized embeddings) |
| `meta.jsonl` | Generated | Varies | Metadata parallel to vecs.npy (chunk ID, text) |
| `bm25.json` | Generated (v2.0) | Varies | BM25 index for keyword search |
| `rag_env/` | N/A | N/A | Python virtual environment (pre-configured) |

### Dependencies

**Python packages** (in `requirements.txt`):
- `requests==2.32.5` – HTTP client for Ollama API calls
- `numpy==2.3.4` – Numerical arrays (embeddings, vectors)

**External services**:
- Ollama (configurable via `OLLAMA_URL` environment variable, defaults to `http://127.0.0.1:11434`) with models:
  - `nomic-embed-text` – 768-dim semantic embeddings
  - `qwen2.5:32b` – LLM for answer generation

**Configuration** (hardcoded in scripts, can be overridden via environment variables):
- `CHUNK_SIZE = 1600` – Characters per chunk
- `CHUNK_OVERLAP = 200` – Character overlap for oversized chunks
- `DEFAULT_TOP_K = 12` (v2.0) – Chunks to retrieve before reranking
- `DEFAULT_PACK_TOP = 6` (v2.0) – Final chunks to include in context
- `DEFAULT_THRESHOLD = 0.30` (v2.0) – Minimum similarity score for acceptance

## Common Development Tasks

### Build the Knowledge Base (One-Time)

**v2.0 (Recommended)**:
```bash
source rag_env/bin/activate
python3 clockify_support_cli_final.py build knowledge_full.md
```
Creates: `chunks.jsonl`, `vecs_n.npy`, `meta.jsonl`, `bm25.json`, `index.meta.json`

**Note**: `clockify_support_cli.py` is a compatibility wrapper that calls `clockify_support_cli_final.py` - both work identically.

**v1.0** (legacy):
```bash
source rag_env/bin/activate
python3 clockify_rag.py chunk
python3 clockify_rag.py embed
```
Creates: `chunks.jsonl`, `vecs.npy`, `meta.jsonl`

### Run the Application

**v2.0 (Interactive REPL)**:
```bash
source rag_env/bin/activate
python3 clockify_support_cli_final.py chat [--debug]
```
Then type questions at the prompt. Type `:exit` to quit, `:debug` to toggle diagnostics.

**v1.0 (Single Query)** (legacy):
```bash
source rag_env/bin/activate
python3 clockify_rag.py ask "Your question here"
```

### Debug/Test an Index

**v2.0 with debug output**:
```bash
python3 clockify_support_cli_final.py chat --debug
> :debug
> Your question
[Shows retrieved chunks, scores, ranking]
```

**Rebuild index** (if corrupted):
```bash
rm -f chunks.jsonl vecs_n.npy meta.jsonl bm25.json index.meta.json
python3 clockify_support_cli_final.py build knowledge_full.md
```

## Thread Safety

**IMPORTANT**: The current implementation is **thread-safe as of v5.1** due to locks added to shared state.

### Deployment Options

**Option 1: Multi-threaded (RECOMMENDED)**
- Deploy with multi-worker, multi-threaded processes (e.g., `gunicorn -w 4 --threads 4`)
- Thread safety locks protect shared state (QueryCache, RateLimiter, _FAISS_INDEX)
- Cache and rate limiter shared across threads within same process

**Option 2: Single-threaded (legacy)**
- Deploy with single-worker processes (e.g., `gunicorn -w 4 --threads 1`)
- Each worker has its own process memory (no shared state)
- Cache and rate limiter per-process

### Thread Safety Implementation

v5.1 adds thread safety through:
- `threading.RLock()` in `QueryCache` (caching.py:92)
- `threading.RLock()` in `RateLimiter` (caching.py:26)
- Double-checked locking for `_FAISS_INDEX` (indexing.py:27-28, clockify_support_cli_final.py:286-287)

All shared state is now protected with reentrant locks to prevent race conditions.

### Thread Safety Validation

Run concurrent tests: `pytest tests/test_thread_safety.py -v -n 4`

## Key Implementation Details

### Chunking Strategy (Both Versions)

- Splits by second-level markdown headings (`##`)
- Enforces max `CHUNK_SIZE` (1600 chars)
- For oversized chunks, creates sub-chunks with `CHUNK_OVERLAP` (200 chars) for context preservation
- Assigns unique monotonic integer IDs

### Retrieval Pipeline

**v1.0**:
1. Embed question with nomic-embed-text
2. Cosine similarity against all chunk vectors
3. Retrieve top 6 chunks
4. Check if ≥2 chunks have similarity ≥0.3
5. Pass to LLM with system prompt

**v2.0 (Hybrid)**:
1. Embed question with nomic-embed-text
2. Retrieve top `DEFAULT_TOP_K` (12) via:
   - **BM25**: Exact keyword matching (sparse)
   - **Dense**: Cosine similarity (semantic)
3. Merge results, apply **MMR** (Maximal Marginal Relevance, lambda=0.7) to diversify
4. Pack top `DEFAULT_PACK_TOP` (6) chunks if similarity ≥ `DEFAULT_THRESHOLD` (0.30)
5. Format snippets with ID, title, context
6. Pass to LLM with system prompt requiring closed-book answers

### LLM Prompting

Both versions instruct the model to:
- Use only provided snippets
- Refuse if information not in snippets
- Return exact phrase: `"I don't know based on the MD."`
- Cite source IDs when answering

### Offline Operation

- All embeddings computed locally (nomic-embed-text)
- All LLM inference local (qwen2.5:32b)
- No external API calls
- No internet required
- Deterministic timeouts (configurable via env vars)

## Testing & Validation

**Example queries** to validate the system:
```bash
python3 clockify_support_cli.py ask "How do I track time in Clockify?"
python3 clockify_support_cli.py ask "What are the pricing plans?"
python3 clockify_support_cli.py ask "Can I track time offline?"
python3 clockify_support_cli.py ask "How do I set up SSO?"
```

**Expected output**:
- Answer with citations: `[id_123, id_456]`
- Or refusal: `"I don't know based on the MD."`

## Configuration & Customization

### Environment Variables (v2.0)

```bash
# Ollama endpoint (default: http://127.0.0.1:11434)
# Override only if Ollama runs on a different machine
export OLLAMA_URL="http://127.0.0.1:11434"

# Model names
export GEN_MODEL="qwen2.5:32b"
export EMB_MODEL="nomic-embed-text"

# Context budget in tokens (~11,200 chars at 4 tokens/word)
export CTX_BUDGET="2800"

# HTTP timeout control
export EMB_CONNECT_TIMEOUT="3"
export EMB_READ_TIMEOUT="120"
export CHAT_CONNECT_TIMEOUT="3"
export CHAT_READ_TIMEOUT="180"
```

### Tuning Parameters (Edit in scripts)

**v1.0** (`clockify_rag.py`, lines 7-15):
```python
CHUNK_SIZE = 1600          # Increase for larger chunks
CHUNK_OVERLAP = 200        # More overlap = better context at boundaries
SIMILARITY_THRESHOLD = 0.3 # Stricter = fewer false positives
```

**v2.0** (`clockify_support_cli.py`, lines 41-51):
```python
CHUNK_CHARS = 1600         # Character limit per chunk
CHUNK_OVERLAP = 200        # Overlap between sub-chunks
DEFAULT_TOP_K = 12         # Retrieve before reranking
DEFAULT_PACK_TOP = 6       # Final chunks in context
DEFAULT_THRESHOLD = 0.30   # Minimum similarity
MMR_LAMBDA = 0.7           # Diversity vs. relevance (0-1)
CTX_TOKEN_BUDGET = 2800    # Context window budget
```

## File Workflows

### Artifact Versioning (v2.0 Only)

`index.meta.json` tracks:
- MD5 hash of knowledge_full.md (detects source changes)
- Last build timestamp
- Index version

If source KB changes, rebuild is automatically triggered.

### Storage Layout

```
/Users/15x/Downloads/KBDOC/
├── clockify_rag.py              (v1.0)
├── clockify_support_cli.py       (v2.0) ← USE THIS
├── knowledge_full.md            (6.9 MB input)
├── chunks.jsonl                 (generated)
├── vecs_n.npy / vecs.npy        (generated)
├── meta.jsonl                   (generated)
├── bm25.json                    (v2.0 only)
├── index.meta.json              (v2.0 only)
├── rag_env/                     (venv)
└── [documentation files]
```

## Troubleshooting Notes for Future Development

### Connection Issues
- Verify Ollama is running: `ollama serve`
- Default endpoint is `http://127.0.0.1:11434` (localhost)
- Override with `OLLAMA_URL` env var if Ollama runs on different machine
- Check connectivity: `curl http://127.0.0.1:11434/api/version`

### Model Issues
- Pull missing models: `ollama pull nomic-embed-text`, `ollama pull qwen2.5:32b`
- Check available: `ollama list`

### Performance Tuning
- **Slow embeddings**: Increase `EMB_READ_TIMEOUT` or reduce `CHUNK_SIZE`
- **Low accuracy**: Adjust `DEFAULT_THRESHOLD` (lower = more lenient), increase `DEFAULT_TOP_K`
- **Memory spikes**: Reduce `DEFAULT_PACK_TOP` or `CTX_TOKEN_BUDGET`

### Build Failures
- **Lock file stale**: Remove `.build.lock` if process crashed
- **Index corrupted**: Delete generated files and rebuild
- **KB changed**: Edit `knowledge_full.md`, trigger rebuild (v2.0 detects automatically)

## Version Comparison

| Aspect | v1.0 | v2.0 |
|--------|------|------|
| Accuracy | ~70% | ~85% |
| Retrieval | Cosine similarity only | Hybrid (BM25 + dense + MMR) |
| UI | Command-line (separate steps) | Interactive REPL |
| Debug capability | None | Yes (`:debug` toggle) |
| Stateful | No | No (each query fresh) |
| Lock mechanism | None | Yes (atomic file creation) |
| File count | 1 script | 1 script |
| Complexity | Low | Moderate |
| **Recommended** | **No** | **YES** ✅ |

## Documentation Map

- **START_HERE.md** – Entry point with quick choice (v1 vs v2)
- **SUPPORT_CLI_QUICKSTART.md** – v2.0 5-minute quick start
- **CLOCKIFY_SUPPORT_CLI_README.md** – v2.0 full technical guide
- **QUICKSTART.md** – v1.0 quick start
- **README_RAG.md** – v1.0 full guide (30 min read)
- **VERSION_COMPARISON.md** – Detailed v1 vs v2 analysis
- **PROJECT_STRUCTURE.md** – Directory layout and file purposes

## Notes for Future Work

1. **Extend knowledge base**: Add new .md sections to `knowledge_full.md`, rebuild index
2. **Switch models**: Change `EMB_MODEL` or `GEN_MODEL` constants, ensure available in Ollama
3. **Deploy to team**: v2.0 is single-file, easier to distribute; include `rag_env/` or require users to install numpy + requests
4. **Optimize retrieval**: v2.0's MMR+BM25 is near optimal; consider cross-encoder reranking for marginal gains
5. **Add feedback loop**: v2.0 doesn't log interactions; could add optional JSON logging for model fine-tuning
6. **Scale to multiple KBs**: Both support single KB only; would need multi-index wrapper

---

## Modular Architecture (v5.0 - NEW!)

**Status**: ✅ Production Ready with Modular Structure & Plugin System

As of v5.0, the system has been refactored into a clean, modular package with plugin architecture:

### Package Structure

```
clockify_rag/                  # Main package (NEW)
├── __init__.py                # Package exports
├── config.py                  # Configuration constants
├── exceptions.py              # Custom exceptions
├── utils.py                   # File I/O, validation, logging
├── http_utils.py              # HTTP session management
├── chunking.py                # Text parsing & chunking
├── embedding.py               # Embeddings (local/Ollama)
├── caching.py                 # Query cache & rate limiting
├── indexing.py                # BM25 & FAISS indexes
└── plugins/                   # Plugin system (NEW)
    ├── __init__.py
    ├── interfaces.py          # Plugin interfaces
    ├── registry.py            # Plugin registry
    └── examples.py            # Example plugins
```

### Using the Modular Package

```python
# Import specific components
from clockify_rag import build, load_index
from clockify_rag.chunking import build_chunks
from clockify_rag.embedding import embed_texts
from clockify_rag.plugins import RetrieverPlugin, register_plugin
```

### Backward Compatibility

The original CLI interface is fully preserved:
```bash
python3 clockify_support_cli.py build knowledge_full.md
python3 clockify_support_cli.py chat --debug
```

### Plugin System

Create custom retrievers, rerankers, embeddings, and indexes:

```python
from clockify_rag.plugins import RetrieverPlugin, register_plugin

class MyRetriever(RetrieverPlugin):
    def retrieve(self, question: str, top_k: int):
        # Custom retrieval logic
        return results

    def get_name(self) -> str:
        return "my_retriever"

register_plugin(MyRetriever())
```

**See [MODULARIZATION.md](MODULARIZATION.md) for complete documentation.**

---

## Recent Improvements (v5.1)

### Bug Fixes (5/5 Critical)

- ✅ **Build lock deadline respected** - Lock now properly times out at 10s (utils.py:148-157, clockify_support_cli_final.py:582-591)
- ✅ **Score normalization fixed** - Preserves rank information when std=0 instead of returning zeros (clockify_support_cli_final.py:1290)
- ✅ **Sliding chunks overlap corrected** - Properly handles overlap at chunk boundaries (chunking.py:103-111, clockify_support_cli_final.py:976-984)
- ✅ **Thread safety implemented (CRITICAL)** - All shared state protected with locks (caching.py, indexing.py, clockify_support_cli_final.py)
- ✅ **Exception handling improved** - Specific error types, preserved tracebacks (clockify_support_cli_final.py:1169-1179)

### Performance Improvements

- ✅ **FAISS index preloaded** at startup - 50-200ms faster first query (clockify_support_cli_final.py:1922-1933)
- ✅ **BM25 early termination threshold lowered** - 2-3x speedup on mid-size corpora (indexing.py:176, clockify_support_cli_final.py:1232)

### Developer Experience

- ✅ **`make dev` target added** - One-command setup (Makefile:112-122)
- ✅ **Cache hit logging** - Improved observability (clockify_support_cli_final.py:2442-2444)

### Reliability & Quality

- ✅ **Thread-safe for multi-threaded deployment** - Can handle concurrent queries safely
- ✅ **Better error messages** - All errors include actionable hints

---

## Recent Improvements (v5.9)

### New in v5.9 (2025-11-08)

Based on comprehensive End-to-End RAG Tool Analysis (RAG_END_TO_END_ANALYSIS.md):

- ✅ **Confidence-based routing** - Auto-escalate low-confidence queries (<40) for human review
- ✅ **Async LLM support** - Optional async/await for 2-4x concurrent throughput improvement
- ✅ **Consolidated tokenization** - Single source of truth for tokenization logic
- ✅ **Enhanced structured logging** - JSON-formatted logs for monitoring dashboard integration
- ✅ **Improved type hints** - Better type safety across codebase

See [IMPROVEMENTS_V5.9.md](IMPROVEMENTS_V5.9.md) for complete details.

---

**Version**: 5.9 (Analysis-Driven Improvements)
**Status**: ✅ Production Ready
**Date**: 2025-11-08
**Platform**: macOS/Linux (v1.0 and v2.0); Windows requires manual venv setup
**Python**: 3.8+
**Apple Silicon**: ✅ M1/M2/M3 Compatible - See [M1_COMPATIBILITY.md](M1_COMPATIBILITY.md) for installation guide
