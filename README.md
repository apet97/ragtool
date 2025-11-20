# 1rag – Clockify Support CLI v5.8

**Status**: ✅ Production Ready (Optimized for Remote Ollama/Qwen Deployments)
**Version**: 5.8 (Configuration Consolidation & Remote Ollama Optimization - 2025-11-08)
**Date**: 2025-11-08

> **Recent**: Consolidated configuration to single source, improved remote Ollama resilience, optimized context budgets for Qwen 32B, enhanced thread safety, and hardened offline deployments. See [CHANGELOG_v5.8.md](CHANGELOG_v5.8.md) for details.

A local, stateless, closed-book Retrieval-Augmented Generation (RAG) chatbot for Clockify support documentation using Ollama.

**New in v5.8**: 🎯 Config consolidation (single source of truth), 🌐 Remote Ollama resilience (retries: 0→2), 🚀 Qwen 32B optimization (context: 2800→6000 tokens), 🔒 Enhanced thread safety, 🔌 Offline-ready (NLTK gated downloads), 🛡️ Query logging security fixes
**New in v5.5**: 🏗️ Removed 186 lines of duplicate code (cache/rate limiter), 📦 Reuse package implementations
**New in v5.4**: 🚀 Optimized query logging (2-3× faster when chunks disabled), 💾 Reduced memory allocation
**New in v5.3**: ⚡ Batched embedding futures (prevents socket exhaustion), 🛡️ Improved stability on large corpora (10,000+ chunks)
**New in v5.2**: 🎯 Deterministic FAISS, 🔐 Enhanced security, 📊 Cache regression tests, 📝 Comprehensive env docs, 🗂️ Clean repository (79% fewer docs)
**New in v5.1**: 🔒 Thread-safe, 🚀 50-200ms faster first query, 🛠️ Better error messages, 🔍 Improved observability

## Platform Compatibility

| Platform | Status | Notes |
|----------|--------|-------|
| **Linux** | ✅ Full Support | Recommended for production |
| **macOS Intel** | ✅ Full Support | Uses IVFFlat FAISS index |
| **macOS Apple Silicon (M1/M2/M3)** | ✅ **Full Support** | See [docs/M1_COMPATIBILITY.md](docs/M1_COMPATIBILITY.md) |
| **Windows** | ⚠️ WSL2 Recommended | Native support via WSL2 |

## Documentation

📚 **Complete documentation index**: [docs/INDEX.md](docs/INDEX.md)

**Quick links by role:**
- **New users**: [START_HERE.md](START_HERE.md) → [SUPPORT_CLI_QUICKSTART.md](SUPPORT_CLI_QUICKSTART.md)
- **Developers**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) → [CLAUDE.md](CLAUDE.md) → [TESTING.md](TESTING.md)
- **DevOps/SRE**: [docs/PRODUCTION_GUIDE.md](docs/PRODUCTION_GUIDE.md) → [docs/SMOKE_TEST_RUNBOOK.md](docs/SMOKE_TEST_RUNBOOK.md)
- **macOS ARM64**: [docs/M1_COMPATIBILITY.md](docs/M1_COMPATIBILITY.md) → [docs/INSTALL_macOS_ARM64.md](docs/INSTALL_macOS_ARM64.md)

## Quick Start

### 5-Step Setup
1. **Clone + install** – `git clone https://github.com/apet97/1rag.git && cd 1rag`, create a virtual env, and run `pip install -e '.[dev]'` (or `pip install .` for runtime-only environments). The legacy `pip install -r requirements.txt` path now maps to `-e .`.
2. **Configure** – copy `.env.example` to `.env`, set `RAG_OLLAMA_URL` (`http://10.127.0.1:11434` locally or `http://10.127.0.192:11434` on the VPN), and adjust timeouts/retries.
3. **Validate workstation** – run `make deps-check` (pip check + targeted pytest) followed by `ragctl doctor --verbose` to confirm dependencies, index readiness, and Ollama connectivity.
4. **Ingest / index** – `make ingest` builds chunks, embeddings, FAISS/BM25 artifacts, and `index.meta.json`.
5. **Smoke test + run** – `make smoke` (mock client by default) followed by `python3 -m clockify_rag.cli_modern chat` or `uvicorn clockify_rag.api:app`.

See `docs/CONFIGURATION.md` for every configurable parameter and `docs/OPERATIONS.md` for day-two operational procedures.

### Installation

**Standard (Linux/macOS Intel)**:
```bash
python3 -m venv rag_env
source rag_env/bin/activate
pip install -e '.[dev]'  # or `pip install .` for runtime-only installs
```

**Apple Silicon (M1/M2/M3) - Recommended**:
```bash
# Use conda for best FAISS compatibility
conda create -n rag_env python=3.11
conda activate rag_env
conda install -c conda-forge faiss-cpu=1.8.0 numpy pandas requests beautifulsoup4 pypdf2 python-docx
conda install -c pytorch pytorch
conda install -c conda-forge sentence-transformers
pip install -e '.[dev]'  # pulls rank-bm25, typer, httpx, etc.
```

For detailed M1 installation instructions, see [docs/M1_COMPATIBILITY.md](docs/M1_COMPATIBILITY.md).

### Configure Ollama Endpoint

**Remote Ollama (Company-Hosted, default)**:
`RAG_OLLAMA_URL` now defaults to the VPN-backed host `http://10.127.0.192:11434`. Ensure you are on the corporate network (or VPN) before running queries.

```bash
# Example: Using remote Ollama server
export RAG_OLLAMA_URL=http://10.127.0.192:11434

# Or inline for a single command
RAG_OLLAMA_URL=http://10.127.0.192:11434 python3 clockify_support_cli_final.py chat
```

**Local Ollama Override**:

```bash
export RAG_OLLAMA_URL=http://127.0.0.1:11434
```

**Timeouts**: Remote endpoints may need higher read timeouts when accessed over VPN:

```bash
export RAG_OLLAMA_URL=http://10.127.0.192:11434
export CHAT_READ_TIMEOUT=300
export EMB_READ_TIMEOUT=180
python3 clockify_support_cli_final.py chat
```

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the complete list of supported environment variables and defaults.

### Build Knowledge Base
```bash
# One-time setup: build the vector index
python3 clockify_support_cli_final.py build knowledge_full.md
```

### Run
```bash
# Start interactive REPL (local Ollama)
python3 clockify_support_cli_final.py chat

# With remote Ollama endpoint
RAG_OLLAMA_URL=http://10.127.0.192:11434 python3 clockify_support_cli_final.py chat

# With debug mode
python3 clockify_support_cli_final.py chat --debug

# Single query (supports --rerank/--json/--topk/--pack)
python3 clockify_support_cli_final.py ask "How do I track time in Clockify?" --rerank --json

# Run self-tests (uses configured RAG_OLLAMA_URL)
python3 clockify_support_cli_final.py --selftest
```

> `ragctl query --json` and `clockify_support_cli_final.py ask --json` emit the same schema as `/v1/query`, including `metadata`, `routing`, and `timing` fields so scripts can automate routing decisions.

### Smoke Test (offline or remote)

```bash
# Deterministic mock client (default)
make smoke

# Retrieval evaluation gate (runs MRR/NDCG thresholds)
make eval-gate

# Dependency + pytest smoke (pip check + targeted tests)
make deps-check

# Full stack against the VPN-hosted Ollama server
RAG_OLLAMA_URL=http://10.127.0.192:11434 \
SMOKE_CLIENT=ollama make smoke
# or run the script directly
python3 scripts/smoke_rag.py --client ollama --question "How do I track time?"

# One-shot verification
make verify
SMOKE_CLIENT=ollama make verify         # run against the VPN-backed host
make verify-ollama                      # convenience alias
```

`make smoke` wraps `scripts/smoke_rag.py`, which now defaults to the deterministic mock LLM client so CI/laptops stay offline. Override with `SMOKE_CLIENT=ollama make smoke` (or run `python3 scripts/smoke_rag.py --client ollama ...`) to exercise the real endpoint when VPN reachability is available. The script loads the local index, runs `answer_once`, and prints routing + latency stats; non-zero exit codes indicate refusal/error so you can fail fast before exposing the API. `make deps-check` is a lightweight dependency health gate (`pip check` + targeted pytest). `make eval-gate` enforces retrieval-quality thresholds (MRR@10/Precision@5/NDCG@10) using `eval_datasets/clockify_v1.jsonl`. `make verify` ties everything together (pip check + quick pytest + smoke + eval). See [VERIFICATION.md](VERIFICATION.md) for the full validation script.

## What's New in v4.1.2

### Apple Silicon M1/M2/M3 Support ✅

**ARM64 Optimizations**:
- ✅ Automatic platform detection (`platform.machine()`)
- ✅ FAISS FlatIP fallback for M1 (prevents segmentation faults)
- ✅ Reduced FAISS cluster count (256→64) for ARM64 stability
- ✅ PyTorch MPS acceleration support (2-3x faster embeddings)
- ✅ 30-70% performance improvement over Intel Macs

**Installation Methods**:
- Conda recommended for M1 (best FAISS compatibility)
- Pip works with automatic fallback if FAISS fails
- Comprehensive troubleshooting guide in [M1_COMPATIBILITY.md](M1_COMPATIBILITY.md)

### Ollama-Only Architecture ✅

**Local Embeddings**:
- ✅ SentenceTransformers integration (`all-MiniLM-L6-v2`)
- ✅ Ollama embeddings via `nomic-embed-text` (fallback/alternative)
- ✅ Configurable via `EMB_BACKEND` environment variable
- ✅ FAISS ANN indexing for fast retrieval (<100ms)

**Hybrid Retrieval**:
- ✅ BM25 sparse retrieval (exact keyword matching)
- ✅ Dense semantic search (cosine similarity)
- ✅ MMR diversification (λ=0.7)
- ✅ Dynamic snippet packing with token budget

**Production Features**:
- ✅ Build lock with TTL (prevents concurrent builds)
- ✅ Artifact versioning (auto-rebuild if KB changes)
- ✅ Atomic file operations (fsync-safe)
- ✅ Graceful FAISS fallback (full-scan if unavailable)
- ✅ Comprehensive logging and KPI metrics

### CLI Enhancements

**Build Command**:
```bash
python3 clockify_support_cli_final.py build knowledge_full.md
```

**Chat Command**:
```bash
python3 clockify_support_cli_final.py chat [--debug] [--rerank]
```

**Ask Command** (single query):
```bash
python3 clockify_support_cli_final.py ask "Your question" [--rerank] [--topk 12] [--pack 6] [--threshold 0.30] [--json]
```

**Self-Test Command**:
```bash
python3 clockify_support_cli_final.py --selftest
```

## Files

### Production Code
- **clockify_support_cli_final.py** – v4.1.2 production script (1,900+ lines)
- **knowledge_full.md** – Clockify documentation knowledge base (~6.9 MB)

### Documentation

**Platform-Specific**
- `M1_COMPATIBILITY.md` – Comprehensive Apple Silicon installation guide
- `M1_COMPATIBILITY_AUDIT.md` – Technical audit report (v4.1.2)
- `M1_COMPREHENSIVE_AUDIT_2025.md` – Complete compatibility analysis

**Getting Started**
- `README.md` (this file) – Quick start and overview
- `CLAUDE.md` – Architecture, common tasks, configuration guide
- `START_HERE.md` – Entry point for new users
- `SUPPORT_CLI_QUICKSTART.md` – 5-minute quick start
- `docs/ARCHITECTURE.md` – entry points, data flow, and component responsibilities
- `docs/OPERATIONS.md` – runbook for rebuilds, deployments, smoke/eval workflows
- `docs/RAG_PROD_CHECKLIST.md` – pre-flight checklist for configuration, ingestion, and ops

**Testing & Validation**
- `scripts/smoke.sh` – Smoke test suite
- `scripts/acceptance_test.sh` – Acceptance tests
- `TEST_GUIDE.md` – Testing documentation

## Acceptance Tests (6/6 Pass)

1. ✅ **Syntax Verification** – Compiles with `python3 -m py_compile`
2. ✅ **Help Flags (Global)** – All 4 timeout flags present
3. ✅ **Help Flags (Chat)** – All 6 new chat flags present
4. ✅ **Config Summary** – Logged at startup with all parameters
5. ✅ **Determinism Check** – SHA256 hashes identical for same question (fixed seed)
6. ✅ **Self-Tests** – 4/4 unit tests pass (MMR sig, pack headroom, RTF guard, float32)

## Key Features

### Security 🔒
- `allow_redirects=False` prevents auth header leaks
- `trust_env=False` by default (set `USE_PROXY=1` to enable)
- All POST calls use explicit (connect, read) timeouts
- Policy guardrails for sensitive queries

### Reliability 🛡️
- urllib3 v1 and v2 compatible retry adapter
- Manual bounded POST retry (max 1 retry, 0.5s backoff)
- Build lock with 10-minute mtime staleness detection
- All writes use atomic fsync-safe operations

### Correctness ✔️
- Deterministic: `temperature=0, seed=42` on all LLM calls
- MMR signature fixed (no missing arguments)
- Headroom enforced: top-1 always included, budget respected
- float32 dtype guaranteed end-to-end

### Observability 👁️
- Config summary at startup
- One-line turn logging with latency metrics
- Rerank fallback logging
- Self-check unit tests

### Privacy Controls 🔐
- `RAG_LOG_INCLUDE_ANSWER` (default `1`): set to `0`, `false`, or `no` to redact raw answers from `rag_queries.jsonl` logs.
- `RAG_LOG_ANSWER_PLACEHOLDER` (optional): customize the placeholder text written when answers are redacted. Leave unset to use `[REDACTED]`; set to an empty string to omit the field entirely.

## Configuration Examples

### Conservative (High Precision)
```bash
python3 clockify_support_cli.py chat \
  --threshold 0.50 \
  --pack 4 \
  --emb-read 180 \
  --chat-read 300
```

### Balanced (Defaults)
```bash
python3 clockify_support_cli.py chat
```

### Aggressive (High Recall)
```bash
python3 clockify_support_cli.py chat \
  --threshold 0.20 \
  --pack 8 \
  --rerank
```

### With Custom Timeouts
```bash
python3 clockify_support_cli.py chat \
  --emb-connect 5 --emb-read 180 \
  --chat-connect 5 --chat-read 300
```

### With Proxy
```bash
USE_PROXY=1 python3 clockify_support_cli.py chat
```

## Testing

### Run Full Test Suite (v5.1+)
```bash
# All tests with coverage
make test

# Or with pytest directly
pytest tests/ --cov=clockify_rag --cov=clockify_support_cli_final --cov-report=html
open htmlcov/index.html
```

### Run Self-Tests
```bash
python3 clockify_support_cli_final.py --selftest
# Expected: [selftest] 4/4 tests passed
```

### Run Thread Safety Tests (v5.1+)
```bash
# Verify thread safety with concurrent queries
pytest tests/test_thread_safety.py -v -n 4

# Integration tests
pytest tests/test_integration.py -v
```

### Run Determinism Check
```bash
python3 clockify_support_cli.py chat --det-check --seed 42
# Expected: [DETERMINISM] ... deterministic=true (both questions)
```

### Run with Debug Logging
```bash
python3 clockify_support_cli.py --log DEBUG chat
```

### Testing and Validation

#### Run All Tests
```bash
# All tests with coverage
make test

# Or with pytest directly
pytest tests/ --cov=clockify_rag --cov-report=html
open htmlcov/index.html
```

#### Run Self-Tests
```bash
python3 clockify_support_cli_final.py --selftest
# Expected: [selftest] 4/4 tests passed
```

#### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/ -k "not integration"

# Integration tests only  
pytest tests/test_integration.py

# Thread safety tests
pytest tests/test_thread_safety.py -n 4
```

For complete testing documentation, see [TESTING.md](TESTING.md).

### Automated Evaluation

Ground-truth retrieval metrics are tracked via `eval.py`. The evaluator works in
both full and lightweight environments:

```bash
# Uses the populated dataset in eval_datasets/
python3 eval.py --dataset eval_datasets/clockify_v1.jsonl

# Enable verbose per-query breakdown
python3 eval.py --dataset eval_datasets/clockify_v1.jsonl --verbose

# Generate an offline LLM answer report (deterministic mock client)
RAG_LLM_CLIENT=mock python3 eval.py --dataset eval_datasets/clockify_v1.jsonl --llm-report
```

Key details:

- ✅ **Hybrid mode** – If the production index artifacts (`chunks.jsonl`,
  `vecs_n.npy`, `bm25.json`) are present, the script evaluates the full hybrid
  retrieval pipeline (`retrieve`).
- ✅ **Lexical fallback** – When embeddings are unavailable (e.g. in CI), the
  script re-chunks `knowledge_full.md` and evaluates using a BM25 index built
  with `rank-bm25`. Only lightweight dependencies are required: `numpy`,
  `requests`, `nltk`, and `rank-bm25`.
- ✅ **Stable relevance labels** – `eval_datasets/clockify_v1.jsonl` stores
  `title` + `section` pairs instead of raw chunk IDs so the dataset remains
  stable across rebuilds.
- ✅ **Thresholds** – Builds fail when any metric drops below the default
  thresholds: `MRR@10 ≥ 0.70`, `Precision@5 ≥ 0.55`, `NDCG@10 ≥ 0.60`.

See [docs/EVALUATION.md](docs/EVALUATION.md) for end-to-end instructions (hybrid vs. lexical, thresholds, and LLM reports) and [eval_datasets/README.md](eval_datasets/README.md) for dataset details.

## Statistics

| Metric | Value |
|--------|-------|
| **Version** | 4.1.2 |
| **Production script** | clockify_support_cli_final.py |
| **Total lines** | ~1,900 |
| **Python version** | 3.9+ (3.11 recommended for M1) |
| **Dependencies** | 7 core + 1 optional (FAISS) |
| **Platform support** | Linux, macOS (Intel + M1), Windows (WSL2) |
| **Knowledge base size** | 6.9 MB (~150 pages) |
| **Embedding dimension** | 384 (all-MiniLM-L6-v2) |
| **Default chunks** | 384 chunks @ 1600 chars each |
| **Build time (M1)** | ~30 seconds |
| **Query latency (M1)** | ~6-11 seconds (LLM dominates) |
| **M1 performance gain** | 30-70% faster than Intel |

## Documentation Map

- `docs/RAG_ARCHITECTURE.md` – system overview and data flow.
- `docs/CONFIGURATION.md` – authoritative list of `RAG_*` and related env vars.
- `docs/API.md` – HTTP contract (same payload as `ragctl query --json`).
- `docs/DEPLOYMENT.md` – platform-specific rollout steps + smoke guidance.
- `docs/RUNBOOK.md` – health checks, troubleshooting, and rebuild procedures.

## Deployment Checklist

### Pre-Deployment
- [ ] Read `README.md` (this file)
- [ ] Read `M1_COMPATIBILITY.md` (if deploying on M1)
- [ ] Read `CLAUDE.md` (architecture overview)
- [ ] Install dependencies (see [Quick Start](#quick-start))
- [ ] Verify platform: `python3 -c "import platform; print(platform.machine())"`

### Installation
- [ ] Create virtual environment (`venv` or `conda`)
- [ ] Install dependencies from `requirements.txt` (or conda)
- [ ] Verify imports: `python3 -m py_compile clockify_support_cli_final.py`

### Build & Test
- [ ] Build knowledge base: `python3 clockify_support_cli_final.py build knowledge_full.md`
- [ ] Run self-tests: `python3 clockify_support_cli_final.py --selftest`
- [ ] Run smoke tests: `make smoke` (Python harness). Use `make smoke-full` for the legacy shell suite.
- [ ] Test query: `python3 clockify_support_cli_final.py ask "How do I track time?"`

### Validation (M1 Only)
- [ ] Verify ARM64 detection in build logs: `"macOS arm64 detected"`
- [ ] Check PyTorch MPS: `python3 -c "import torch; print(torch.backends.mps.is_available())"`
- [ ] Run M1 compatibility tests: `bash scripts/m1_compatibility_test.sh` (if available)

### Production (v5.1+)
- [ ] Configure Ollama endpoint (`RAG_OLLAMA_URL` env var; `OLLAMA_URL` still accepted)
- [ ] Set production timeouts (if needed)
- [ ] Test end-to-end query
- [ ] **Choose deployment mode** (see below)
- [ ] Deploy to production
- [ ] Monitor first queries for errors

#### Thread-Safe Deployment (v5.1+)

**Option 1: Multi-threaded (RECOMMENDED)**
```bash
# Deploy with multi-worker, multi-threaded processes
gunicorn -w 4 --threads 4 app:app

# Or with uvicorn
uvicorn app:app --workers 4
```
- Thread safety locks protect shared state (QueryCache, RateLimiter, _FAISS_INDEX)
- Cache and rate limiter shared across threads within same process
- Better resource utilization

**Option 2: Single-threaded (legacy)**
```bash
# Deploy with single-worker processes
gunicorn -w 4 --threads 1 app:app
```
- Each worker has its own process memory (no shared state)
- Cache and rate limiter per-process

## Documentation Reading Paths

### 🚀 For Immediate Deployment (5 min)
1. `README.md` (this file) – Quick start
2. Install dependencies (see [Quick Start](#quick-start))
3. Build & test: `python3 clockify_support_cli_final.py build knowledge_full.md`
4. Run: `python3 clockify_support_cli_final.py chat`

### 🍎 For Apple Silicon M1/M2/M3 (10 min)
1. `M1_COMPATIBILITY.md` – Installation guide
2. `M1_COMPREHENSIVE_AUDIT_2025.md` – Full compatibility analysis
3. Use conda for installation (recommended)

### 🏗️ For Architecture (30 min)
1. `CLAUDE.md` – Architecture & development guide
2. `clockify_support_cli_final.py` – Read code comments
3. `M1_COMPATIBILITY.md` – Platform-specific optimizations

### ✅ For Testing & Validation (15 min)
1. Run: `python3 clockify_support_cli_final.py --selftest`
2. Run: `bash scripts/smoke.sh`
3. Run: `bash scripts/acceptance_test.sh`

## Requirements

### Core Dependencies
- **Python 3.9+** (3.11 recommended for M1 Macs)
- **numpy** 2.3.4 – Vector operations and embeddings
- **requests** 2.32.5 – HTTP client for Ollama API
- **urllib3** 2.2.3 – Low-level HTTP
- **sentence-transformers** 3.3.1 – Local embedding generation
- **torch** 2.4.2 – PyTorch for neural models
- **rank-bm25** 0.2.2 – Sparse retrieval (BM25 algorithm)

### Optional Dependencies
- **faiss-cpu** 1.8.0.post1 – Fast approximate nearest neighbors (ANN)
  - Required for optimal performance
  - **M1 Users**: Install via conda for best compatibility
  - Graceful fallback to full-scan if unavailable

### External Services
- **Ollama** – Remote/Local LLM server
  - Default endpoint: `http://10.127.0.192:11434` (VPN). Override to `http://127.0.0.1:11434` for local testing.
  - Required models:
    - `nomic-embed-text` – 768-dim embeddings (optional if using local)
    - `qwen2.5:32b` – LLM for answer generation

### Installation
See [Quick Start](#quick-start) section above or [M1_COMPATIBILITY.md](M1_COMPATIBILITY.md) for M1-specific instructions.

## Environment Variables

### Core Configuration
```bash
RAG_OLLAMA_URL          # Ollama endpoint (default: http://10.127.0.192:11434; override to http://127.0.0.1:11434 locally)
RAG_CHAT_MODEL          # Generation model (default: qwen2.5:32b)
RAG_EMBED_MODEL         # Embedding model (default: nomic-embed-text:latest)
EMB_BACKEND             # "local" or "ollama" (default: local)
```

Legacy aliases (`OLLAMA_URL`, `GEN_MODEL`, `EMB_MODEL`) remain supported but will be removed in a future major release.

Set `RAG_LLM_CLIENT=mock` to force the deterministic offline client (used automatically in CI/tests). Any other value uses the real Ollama server.

### Performance Tuning
```bash
CTX_BUDGET              # Context token budget (default: 2800)
ANN                     # ANN backend: "faiss" or "none" (default: faiss)
ANN_NLIST               # FAISS clusters (default: 64 for M1, 256 otherwise)
ANN_NPROBE              # FAISS search clusters (default: 16)
ALPHA                   # Hybrid retrieval weight (default: 0.5)
FAISS_CANDIDATE_MULTIPLIER  # ANN candidate multiplier (default: 3)
```

### Timeouts
```bash
EMB_CONNECT_TIMEOUT     # Embedding connect timeout (default: 3)
EMB_READ_TIMEOUT        # Embedding read timeout (default: 60)
CHAT_CONNECT_TIMEOUT    # Chat connect timeout (default: 3)
CHAT_READ_TIMEOUT       # Chat read timeout (default: 120)
RERANK_READ_TIMEOUT     # Rerank timeout (default: 180)
```

### Query Logging
```bash
RAG_LOG_FILE            # Log file path (default: rag_queries.jsonl)
RAG_NO_LOG              # Disable logging: 1/0 (default: 0)
RAG_LOG_INCLUDE_ANSWER  # Include answer text: 1/0 (default: 1)
RAG_LOG_ANSWER_PLACEHOLDER  # Placeholder when answer redacted (default: [REDACTED])
RAG_LOG_INCLUDE_CHUNKS  # Include chunk text: 1/0 (default: 0 for security)
```
See [LOGGING_CONFIG.md](LOGGING_CONFIG.md) for comprehensive logging documentation.

### Caching & Rate Limiting
```bash
CACHE_MAXSIZE           # Query cache size (default: 100)
CACHE_TTL               # Cache TTL in seconds (default: 3600)
RATE_LIMIT_REQUESTS     # Max requests per window (default: 10)
RATE_LIMIT_WINDOW       # Window in seconds (default: 60)
```

### Query Expansion
```bash
CLOCKIFY_QUERY_EXPANSIONS   # Path to custom query_expansions.json
MAX_QUERY_EXPANSION_FILE_SIZE  # Max file size in bytes (default: 10485760 / 10MB)
```

### Build & Warm-up
```bash
BUILD_LOCK_TTL_SEC      # Build lock timeout (default: 900)
WARMUP                  # Enable warm-up: 1/0 (default: 1)
```

### M1-Specific
```bash
# Force FAISS fallback (useful if FAISS crashes on M1)
export USE_ANN=none

# Enable PyTorch MPS fallback
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Example Configurations

**Production (Privacy-Preserving)**:
```bash
export RAG_LOG_FILE="/var/log/rag/queries.jsonl"
export RAG_LOG_INCLUDE_ANSWER=0
export RAG_LOG_INCLUDE_CHUNKS=0
export CACHE_MAXSIZE=500
export CACHE_TTL=7200
```

**Development (Full Logging)**:
```bash
export RAG_LOG_FILE="dev_queries.jsonl"
export RAG_LOG_INCLUDE_ANSWER=1
export RAG_LOG_INCLUDE_CHUNKS=1
export WARMUP=0
```

**Performance Tuning**:
```bash
export CTX_BUDGET=4000
export FAISS_CANDIDATE_MULTIPLIER=5
export CACHE_MAXSIZE=200
```

## Production Status

✅ **READY FOR IMMEDIATE DEPLOYMENT**

**v4.1.2 Features**:
- ✅ Full M1/M2/M3 Apple Silicon support
- ✅ Ollama-optimized architecture
- ✅ Local embeddings (SentenceTransformers)
- ✅ Hybrid retrieval (BM25 + dense + MMR)
- ✅ FAISS ANN indexing with graceful fallback
- ✅ Comprehensive testing suite
- ✅ Production-grade error handling
- ✅ Platform-specific optimizations

**Compatibility**:
- ✅ Backward compatible with v3.x
- ✅ Works on Linux, macOS (Intel + M1), Windows (WSL2)
- ✅ Graceful degradation (FAISS optional)
- ✅ 30-70% performance improvement on M1

## Support

For questions about:
- **Quick Start**: See [Quick Start](#quick-start) above
- **M1 Installation**: See `M1_COMPATIBILITY.md`
- **Architecture**: See `CLAUDE.md`
- **Testing**: Run `python3 clockify_support_cli_final.py --selftest`
- **Troubleshooting**: See `M1_COMPATIBILITY.md` (troubleshooting section)

### Common Issues

**FAISS import error on M1**:
```bash
# Solution: Use conda
conda install -c conda-forge faiss-cpu=1.8.0
```

**Slow performance**:
```bash
# Check if running native ARM (not Rosetta)
python3 -c "import platform; print(platform.machine())"
# Expected on M1: arm64
```

**Missing dependencies**:
```bash
pip install -r requirements.txt
# Or for M1: see M1_COMPATIBILITY.md
```

---

**Version**: 4.1.2 (Ollama-Optimized with M1 Support)
**Date**: 2025-11-05
**Status**: 🚀 **PRODUCTION-READY**
**Platform**: Linux | macOS (Intel + Apple Silicon) | Windows (WSL2)
