# Clockify RAG - Production-Ready System

**Version:** 5.9.1
**Status:** ✅ Production Ready
**Platform Support:** macOS (ARM64/Intel), Linux (x86-64/ARM64), Docker (multi-arch)

A production-grade Retrieval-Augmented Generation (RAG) system for Clockify documentation with:

- ✅ **Hybrid Retrieval**: BM25 (keyword) + Dense (semantic) + MMR (diversity)
- ✅ **Apple Silicon Support**: Optimized for M1/M2/M3 Macs with MPS acceleration
- ✅ **Fully Offline**: All processing local (no external API calls)
- ✅ **Multi-Backend**: FAISS, HNSW, or BM25-only for resource-constrained environments
- ✅ **Fast**: Sub-second queries with intelligent caching
- ✅ **Modular**: Clean architecture, easy to extend
- ✅ **Well-Tested**: Unit tests, integration tests, smoke tests
- ✅ **Docker-Ready**: Multi-platform images (linux/amd64, linux/arm64)
- ✅ **Type-Safe**: Full type hints, mypy compatible

## 🚀 Quick Start (90 seconds)

### macOS (ARM64/Intel)

```bash
# 1. Clone and setup
git clone https://github.com/apet97/1rag.git
cd 1rag
bash scripts/bootstrap_macos_arm64.sh

# 2. Build index
python3 -m clockify_rag.cli_modern ingest --input knowledge_full.md

# 3. Start interactive chat
python3 -m clockify_rag.cli_modern chat
> What is Clockify?
> How do I track time offline?
> :exit
```

### Linux

```bash
# Install Python 3.11
apt-get update && apt-get install -y python3.11 python3.11-venv

# Setup virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# Build index and chat
python3 -m clockify_rag.cli_modern ingest --input knowledge_full.md
python3 -m clockify_rag.cli_modern chat
```

### Docker

```bash
# Start API server with Ollama
docker-compose --profile with-ollama up -d

# Check health
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I track time?"}'
```

## 📋 Commands

### Doctor (System Check)

```bash
# Quick diagnostics
ragctl doctor

# JSON output for scripting
ragctl doctor --json | jq .device
```

### Build Index

```bash
# From markdown file
ragctl ingest --input knowledge_full.md

# Force rebuild
ragctl ingest --force

# From directory (finds *.md files)
ragctl ingest --input ./docs
```

### Query (Single)

```bash
# Simple query
ragctl query "How do I track time in Clockify?"

# With parameters
ragctl query "Question" --top-k 20 --pack-top 10 --threshold 0.2

# JSON output
ragctl query "Question" --json | jq .confidence
```

### Chat (Interactive)

```bash
# Start interactive session
ragctl chat

# With custom parameters
ragctl chat --top-k 20 --debug
```

### API Server

```bash
# Start REST API (port 8000)
python3 -m uvicorn clockify_rag.api:app --host 0.0.0.0 --port 8000

# Or with gunicorn for production
gunicorn -w 4 --bind 0.0.0.0:8000 clockify_rag.api:app
```

### Rate Limiting / Throttling

Every inbound question flows through a sliding-window rate limiter. Operations teams can tune it with two
environment variables:

| Variable | Description |
| --- | --- |
| `RATE_LIMIT_REQUESTS` | Number of requests allowed per identity within the window (default: `10`). |
| `RATE_LIMIT_WINDOW` | Length of the sliding window in seconds (default: `60`). |

Identities are derived from the CLI process (per PID) or from API credentials / client IPs. When the limit is
exceeded, CLI users see a friendly "try again later" prompt while API callers receive HTTP 429 responses with a
`Retry-After` hint. Setting either variable to `0` disables throttling if you have dedicated infrastructure.

## 📊 System Architecture

```
Knowledge Base (markdown)
        ↓
    Chunking (semantic splits)
        ↓
    Embedding (local SentenceTransformer)
        ↓
  ┌─ FAISS Index (ANN vector search)
  ├─ HNSW Index (ANN fallback)
  └─ BM25 Index (keyword search)
        ↓
   Retrieval (hybrid BM25+dense+MMR)
        ↓
   LLM Inference (Ollama local or API)
        ↓
     Answer + Citations
```

## 🔧 Configuration

See [docs/CONFIG.md](CONFIG.md) for detailed parameter documentation.

Key environment variables:
- `OLLAMA_URL`: Ollama service URL (default: http://127.0.0.1:11434)
- `GEN_MODEL`: Generation model (default: qwen2.5:32b)
- `EMB_MODEL`: Embedding model (default: nomic-embed-text)
- `DEFAULT_TOP_K`: Retrieval candidates (default: 15)
- `DEFAULT_PACK_TOP`: Final context chunks (default: 8)
- `DEFAULT_THRESHOLD`: Minimum similarity (default: 0.25)

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [INSTALL_macOS_ARM64.md](INSTALL_macOS_ARM64.md) | Apple Silicon setup guide |
| [CONFIG.md](CONFIG.md) | All configuration options |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design and components |
| [API.md](API.md) | REST API reference |
| [OPERATIONS.md](OPERATIONS.md) | Deployment and monitoring |

## 🐳 Docker

### Single Service (API only)

```bash
docker build -t clockify-rag:latest .
docker run -p 8000:8000 \
  -e OLLAMA_URL=http://host.docker.internal:11434 \
  clockify-rag:latest
```

### Full Stack (with Ollama)

```bash
docker-compose --profile with-ollama up -d
curl http://localhost:8000/v1/health
```

### Multi-Architecture Build

```bash
# Build and push for amd64 + arm64
docker buildx build --platform linux/amd64,linux/arm64 \
  -t myrepo/clockify-rag:latest --push .
```

## 🧪 Testing

```bash
# Unit tests
pytest tests/ -v

# Coverage report
pytest tests/ --cov=clockify_rag --cov-report=html

# Integration tests (requires Ollama)
pytest tests/test_integration.py -v

# Smoke test (simple query)
python3 -m clockify_rag.cli_modern query "test"
```

## 📈 Performance

### Benchmarks (M1 Pro, 16GB)

- **Index Build**: ~30 seconds (384 chunks)
- **First Query**: ~2 seconds (model warmup)
- **Subsequent Queries**: ~1 second (with cache)
- **Embedding Generation**: 100 texts/minute (multi-threaded)
- **Memory Usage**: ~1.2 GB peak (with models loaded)

### Optimization Tips

1. **Reduce TOP_K** for faster retrieval (trade-off: recall)
2. **Enable FAQ cache** for common questions (requires precomputation)
3. **Use BM25-only mode** on resource-constrained systems
4. **Enable query caching** for repeated questions
5. **Batch queries** via API for throughput

## 🔐 Security Considerations

- ✅ All processing local (no external API calls for embedding/generation)
- ✅ No API keys required (optional for external LLM providers)
- ✅ Input validation and sanitization
- ✅ No query logging by default (opt-in via RAG_LOG_FILE)
- ✅ Non-root Docker user (raguser)
- ⚠️ CORS enabled in API by default (customize for production)

## 🆘 Troubleshooting

### macOS: "Wrong architecture (x86_64, need arm64)"

You're running under Rosetta (x86 emulation). Fix:

```bash
# Verify
python3 -c "import platform; print(platform.machine())"  # Should show 'arm64'

# Reinstall Python
brew uninstall python@3.11
brew install python@3.11  # Native ARM64 from Homebrew

# Verify pyenv
which python3  # Should be /opt/homebrew/bin/python3
```

### Ollama Connection Failed

```bash
# Check if Ollama is running
ollama serve

# Check models
ollama list

# Pull missing models
ollama pull nomic-embed-text
ollama pull qwen2.5:32b

# Test connectivity
curl http://127.0.0.1:11434/api/version
```

### Index Build Fails

```bash
# Check knowledge base exists
ls -lh knowledge_full.md

# Force rebuild (remove artifacts)
rm -f chunks.jsonl vecs_n.npy meta.jsonl bm25.json index.meta.json

# Retry
ragctl ingest --input knowledge_full.md --force
```

### Low Answer Quality

1. **Increase TOP_K**: `DEFAULT_TOP_K=20 ragctl query "question"`
2. **Lower threshold**: `DEFAULT_THRESHOLD=0.15 ragctl query "question"`
3. **Check source documents**: Are answers in knowledge_full.md?
4. **Enable debug**: `ragctl query "q" --debug` to see retrieval scores
5. **Verify embeddings**: Check if embedding model is appropriate for domain

## 📦 Requirements

- **Python**: 3.11+ (3.12 supported)
- **Memory**: 4GB minimum, 8GB recommended (with models)
- **Disk**: 2GB for index artifacts, 10GB+ for Ollama models
- **CPU**: Any modern CPU (M1+ recommended for macOS)
- **GPU**: Optional (MPS on macOS, CUDA on Linux)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure `black`, `ruff`, and `mypy` pass
5. Open a pull request

## 📄 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

- **Sentence Transformers** for embedding models
- **FAISS** for vector indexing
- **Ollama** for local LLM inference
- **FastAPI** and **Typer** for API and CLI frameworks

## 📞 Support

- 📖 **Documentation**: See `/docs` folder
- 🐛 **Issues**: GitHub Issues
- 💬 **Discussions**: GitHub Discussions
- 📧 **Email**: support@clockify.me

---

**Ready to get started?** Follow the [Quick Start](#-quick-start-90-seconds) above or read [INSTALL_macOS_ARM64.md](INSTALL_macOS_ARM64.md) for detailed instructions.
