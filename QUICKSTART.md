# Clockify RAG CLI – Quick Start Guide

**Version**: 5.5 | **Status**: ✅ Production Ready

> **Priority #20**: Single authoritative quickstart (consolidated from multiple guides)

This guide gets you up and running with the Clockify Support CLI in 5-10 minutes.

---

## Prerequisites

- Python 3.7+ installed
- [Ollama](https://ollama.ai) running locally at `http://127.0.0.1:11434`

---

## Installation

### Option 1: Standard Setup (Linux, macOS Intel, WSL2)

```bash
# Clone or navigate to project directory
cd /path/to/1rag

# Create virtual environment
python3 -m venv rag_env
source rag_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Apple Silicon (M1/M2/M3) – Recommended

For best FAISS compatibility on Apple Silicon, use conda:

```bash
# Create conda environment
conda create -n rag_env python=3.11
conda activate rag_env

# Install FAISS and dependencies
conda install -c conda-forge faiss-cpu=1.8.0 numpy requests
conda install -c pytorch sentence-transformers pytorch
pip install urllib3==2.2.3 rank-bm25==0.2.2
```

> **Note**: See [M1_COMPATIBILITY.md](M1_COMPATIBILITY.md) for detailed Apple Silicon instructions and troubleshooting.

---

## Setup Ollama

### Local Ollama (Default)

Ensure Ollama is running locally and has the required models:

```bash
# Start Ollama server (in separate terminal)
ollama serve

# Verify Ollama is accessible
curl http://127.0.0.1:11434/api/version

# Pull required models
ollama pull nomic-embed-text    # For embeddings (768-dim)
ollama pull qwen2.5:32b          # For answer generation
```

### Remote Ollama (Company-Hosted)

If you're using a company-hosted Ollama endpoint (e.g., `http://10.127.0.192:11434`), you don't need to run `ollama serve` locally. Instead, configure the endpoint:

```bash
# Set the remote Ollama URL
export OLLAMA_URL=http://10.127.0.192:11434

# Verify remote Ollama is accessible
curl $OLLAMA_URL/api/version

# Check available models on remote server
curl $OLLAMA_URL/api/tags
```

**Important**: Remote endpoints may require increased timeouts, especially over VPN:

```bash
export OLLAMA_URL=http://10.127.0.192:11434
export CHAT_READ_TIMEOUT=300
export EMB_READ_TIMEOUT=180
```

---

## Build Knowledge Base (One-Time)

This step processes the Clockify documentation and creates searchable indexes (~5-10 minutes):

```bash
# Activate environment (if not already active)
source rag_env/bin/activate   # or: conda activate rag_env

# Build index from knowledge base
python3 clockify_support_cli_final.py build knowledge_full.md
```

**Expected output**:
```
======================================================================
BUILDING KNOWLEDGE BASE
======================================================================

[1/4] Parsing and chunking...
  Created 1247 chunks

[2/4] Embedding with local (backend=local)...
  Cache: 0/1247 hits (0.0%)
  Computing 1247 new embeddings...
  Progress: ████████████████████ 100%
  Saved (1247, 768) embeddings (normalized)

[3/4] Building BM25 index...
  Indexed 12840 unique terms

[4/4] Building FAISS index...
  Training IVF clusters...
  FAISS index saved: faiss.index

======================================================================
BUILD COMPLETE
======================================================================
Generated files:
  - chunks.jsonl (1247 chunks)
  - vecs_n.npy (normalized embeddings)
  - bm25.json (keyword index)
  - faiss.index (approximate nearest neighbor index)
  - meta.jsonl (chunk metadata)
  - emb_cache.jsonl (embedding cache for incremental builds)
```

> **Note**: Subsequent builds are faster thanks to embedding cache (only re-processes changed chunks).

---

## Run the Application

### Interactive REPL (Recommended)

**Local Ollama**:
```bash
python3 clockify_support_cli_final.py chat
```

**Remote Ollama** (company-hosted endpoint):
```bash
OLLAMA_URL=http://10.127.0.192:11434 python3 clockify_support_cli_final.py chat
```

**REPL commands**:
- Type questions naturally, press Enter
- `:exit` – quit the REPL
- `:debug` – toggle debug mode (shows retrieval details)
- `:help` – show available commands

**Example session**:
```
Clockify Support CLI v5.5 - Type :exit to quit, :debug for diagnostics
Using Ollama at: http://10.127.0.192:11434

> How do I track time in Clockify?

To track time in Clockify, you have two main options:

1. **Timer**: Click the timer button in the top right corner to start tracking time in real-time
2. **Manual entry**: Add time entries afterward using the manual time entry form

You can track time in the web app, desktop app, mobile app, or browser extension.

Citations: [id_45, id_67, id_89]

> :exit
Goodbye!
```

### Single Query Mode

```bash
# Ask a single question
python3 clockify_support_cli_final.py ask "How do I track time?"

# With additional options
python3 clockify_support_cli_final.py ask "What are the pricing plans?" \
  --rerank \      # Use LLM-based reranking
  --json \        # Output as JSON
  --topk 15 \     # Retrieve more candidates
  --pack 8        # Include more snippets in context
```

### Debug Mode

```bash
# Start REPL with debug mode enabled
python3 clockify_support_cli_final.py chat --debug

# Or toggle debug during session
> :debug
Debug mode: ON
```

Debug mode shows:
- Retrieved chunk IDs and scores
- Dense, BM25, and hybrid scores
- MMR diversification results
- Token usage and performance metrics

---

## Configuration

### Environment Variables

Common configuration options:

```bash
# Ollama endpoint (if running remotely)
export OLLAMA_URL="http://10.0.0.5:11434"

# Model selection
export GEN_MODEL="qwen2.5:32b"        # Answer generation
export EMB_MODEL="nomic-embed-text"   # Embeddings

# Retrieval tuning
export ALPHA="0.5"                    # BM25/dense blend (0=dense, 1=BM25)
export CTX_BUDGET="2800"              # Token budget for context

# Embedding backend
export EMB_BACKEND="local"            # "local" or "ollama"

# Performance
export ANN="faiss"                    # "faiss" or "none"
export ANN_NLIST="64"                 # FAISS IVF clusters
export ANN_NPROBE="16"                # FAISS search clusters

# Privacy
export RAG_NO_LOG="1"                 # Disable query logging
export RAG_LOG_INCLUDE_ANSWER="0"    # Redact answers from logs
```

### Tuning Retrieval Parameters

Adjust retrieval behavior via command-line flags:

```bash
# Retrieve more candidates (default: 12)
python3 clockify_support_cli_final.py chat --topk 20

# Include more snippets in context (default: 6)
python3 clockify_support_cli_final.py chat --pack 10

# Adjust similarity threshold (default: 0.30)
python3 clockify_support_cli_final.py chat --threshold 0.40

# Enable LLM-based reranking (slower, higher quality)
python3 clockify_support_cli_final.py chat --rerank
```

---

## Verification

Run self-tests to verify installation:

```bash
python3 clockify_support_cli_final.py --selftest
```

Run evaluation suite (requires eval dataset):

```bash
# Run full evaluation
./eval.py --verbose

# Quick evaluation (lexical fallback if no index)
./eval.py
```

---

## Troubleshooting

### Connection Errors

**Problem**: `Connection refused` or `EmbeddingError`

**Solution**:
1. Verify Ollama is running: `curl http://127.0.0.1:11434/api/version`
2. Check model availability: `ollama list`
3. Pull missing models: `ollama pull nomic-embed-text`

### Build Fails

**Problem**: Build crashes or hangs

**Solution**:
1. Remove lock file: `rm .build.lock`
2. Clear corrupted artifacts: `rm -f chunks.jsonl vecs_n.npy bm25.json faiss.index`
3. Rebuild: `python3 clockify_support_cli_final.py build knowledge_full.md`

### Apple Silicon Issues

**Problem**: Segmentation fault or FAISS errors on M1/M2/M3

**Solution**: See [M1_COMPATIBILITY.md](M1_COMPATIBILITY.md) for:
- Conda-based installation (recommended)
- FAISS fallback to FlatIP
- PyTorch MPS acceleration

### Poor Answer Quality

**Problem**: Answers are inaccurate or incomplete

**Solution**:
1. Enable debug mode to inspect retrieved chunks: `--debug`
2. Increase candidates: `--topk 20`
3. Adjust BM25/dense blend: `export ALPHA="0.3"` (more semantic) or `export ALPHA="0.7"` (more keyword-based)
4. Enable reranking: `--rerank`

---

## Next Steps

- **Production Deployment**: See [CLAUDE.md](CLAUDE.md) for thread-safety, caching, and deployment options
- **Architecture Details**: See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for codebase organization
- **API Integration**: Import `clockify_rag` package for programmatic access
- **Customization**: Edit `config/query_expansions.json` for domain-specific synonyms

---

## Support

For issues, questions, or contributions:
- Check [CLAUDE.md](CLAUDE.md) for detailed architecture documentation
- Review [VERSION_COMPARISON.md](VERSION_COMPARISON.md) for v1.0 vs v2.0 differences
- See [CI_CD_M1_RECOMMENDATIONS.md](CI_CD_M1_RECOMMENDATIONS.md) for CI/CD setup

---

**Last Updated**: 2025-11-07 | **Version**: 5.5
