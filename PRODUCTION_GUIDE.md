# Production Deployment Guide

## Overview
This document provides guidance for deploying the RAG system in production environments with optimal performance, reliability, and security.

## Architecture Overview

### Core Components
- **Configuration Layer**: Centralized configuration with validation and environment variable management
- **Ingestion Layer**: Document parsing and normalization across multiple formats
- **Indexing Layer**: Vector and lexical indexing with FAISS and BM25
- **Retrieval Layer**: Hybrid search with semantic and lexical components
- **Generation Layer**: LLM-based answer generation with citation validation
- **API Layer**: Ollama-compatible client with retry logic and error handling

### Data Flow
```
Document Sources → Ingestion → Chunking → Embedding → Indexing → Query → Retrieval → Generation → Response
```

## Production Configuration

### Critical Dependencies

**IMPORTANT**: The system requires `langchain-ollama` in production environments.

```bash
# Production (REQUIRED)
pip install langchain-ollama

# Development (optional fallback)
pip install langchain-community
```

**Why this matters:**
- `langchain-ollama` is the newer, better-maintained package for Ollama integration
- `langchain-community` is deprecated for new deployments
- Production deployments **MUST NOT** rely on silent fallback to legacy packages

**Environment Control:**
```bash
# Set ENVIRONMENT to control import behavior
export ENVIRONMENT=production  # Fail fast if langchain-ollama missing
export ENVIRONMENT=dev         # Allow fallback to langchain-community (not recommended)
export ENVIRONMENT=ci          # Same as production (fail fast)

# Alternative: APP_ENV (checked if ENVIRONMENT not set)
export APP_ENV=production
```

### Required Environment Variables
```bash
# Environment Type (CRITICAL)
ENVIRONMENT=production                   # Controls dependency enforcement (prod/ci/dev)

# Ollama Configuration
OLLAMA_URL=http://your-ollama-host:11434  # Your Ollama endpoint
GEN_MODEL=qwen2.5:32b                     # Generation model
EMB_MODEL=nomic-embed-text               # Embedding model

# Performance Tuning
CTX_BUDGET=12000                         # Context token budget
DEFAULT_TOP_K=15                         # Retrieval depth (default candidates)
MAX_TOP_K=50                             # Maximum top-K ceiling (safety cap)
DEFAULT_PACK_TOP=8                       # Snippets to pack in context

# Security and Privacy
RAG_LOG_INCLUDE_ANSWER=1                 # Include answers in logs (0 for privacy)
RAG_LOG_INCLUDE_CHUNKS=0                 # Include chunk text (0 for security)

# Connection Settings
CHAT_READ_TIMEOUT=180                    # LLM call timeout
EMB_READ_TIMEOUT=120                     # Embedding timeout
DEFAULT_RETRIES=2                        # Retry attempts for transient errors
```

### Optional Configuration
```bash
# Advanced Settings
MMR_LAMBDA=0.75                          # MMR diversity vs relevance balance
ALPHA=0.5                                # BM25 vs dense score blend
ANN=faiss                                # Use FAISS for ANN (or "none" for linear)
USE_INTENT_CLASSIFICATION=1              # Enable intent-aware retrieval

# Resource Management
CACHE_MAXSIZE=500                        # Query cache size
CACHE_TTL=7200                           # Cache TTL in seconds
EMB_MAX_WORKERS=8                        # Parallel embedding workers

# Operational
WARMUP=1                                 # Preload models on startup
LOG_LEVEL=INFO                           # Logging level
```

## Deployment Options

### Standalone Deployment
```bash
# 1. Clone repository
git clone <repo-url>
cd 1rag

# 2. Set up Python environment
python3 -m venv rag_env
source rag_env/bin/activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your settings

# 4. Build knowledge base
python clockify_support_cli_final.py build knowledge_full.md

# 5. Start service
python clockify_support_cli_final.py chat
```

### Docker Deployment
```bash
# Build and start with Docker Compose
docker-compose up -d --build

# Or run standalone container
docker build -t rag-service .
docker run -d -p 8000:8000 \
  -e OLLAMA_URL=http://your-ollama:11434 \
  -v ./knowledge:/app/knowledge \
  rag-service
```

### systemd Service (Linux)
Create `/etc/systemd/system/rag-service.service`:
```ini
[Unit]
Description=RAG Service
After=network.target

[Service]
Type=simple
User=rag-user
WorkingDirectory=/path/to/rag
Environment=OLLAMA_URL=http://127.0.0.1:11434
ExecStart=/path/to/venv/bin/python clockify_support_cli_final.py chat
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable rag-service
sudo systemctl start rag-service
```

## Performance Optimization

### Indexing Performance
- Use FAISS for large corpora (>1000 documents)
- Enable parallel embedding with `EMB_MAX_WORKERS`
- Pre-compute embeddings with caching to avoid recomputation

### Query Performance
- Configure ANN settings appropriately for your data size
- Tune `DEFAULT_TOP_K` and `DEFAULT_PACK_TOP` for optimal latency
- Use warm-up to pre-load models: `WARMUP=1`

### Memory Management
- Monitor embedding cache size with `CACHE_MAXSIZE`
- Consider using float16 for embeddings to reduce memory usage
- Set appropriate timeouts to prevent connection leaks

## Security Considerations

### Network Security
- Use TLS for connections to Ollama if not on localhost
- Configure proper firewall rules for Ollama access
- Implement network segmentation between RAG service and LLM backend

### Data Security
- Redact sensitive data from logs using `RAG_LOG_INCLUDE_CHUNKS=0`
- Use secure temporary files for processing
- Validate all user inputs to prevent injection attacks

### Access Control
- Implement authentication/authorization layer in front of API
- Use environment-specific API keys if required
- Rotate Ollama endpoints periodically

## Monitoring and Observability

### Log Management
- Configure structured logging with JSON format
- Set up log rotation to prevent disk space issues
- Monitor key performance metrics in logs

### Health Checks
```bash
# Check environment
python scripts/verify_env.py

# Verify fallback behavior (model selection)
python scripts/verify_fallback.py

# Check system health
python -c "from clockify_rag.error_handlers import print_system_health; print_system_health()"

# Run sanity check (end-to-end)
python -m clockify_rag.sanity_check

# Run self-tests
python clockify_support_cli_final.py --selftest
```

### Key Metrics to Monitor
- Query latency and throughput
- Cache hit rates
- Model error rates
- Index rebuild frequency

## Troubleshooting

### Common Issues

#### Ollama Connection
- Verify Ollama is running: `curl http://host:11434/api/tags`
- Check model availability: `ollama list`
- Verify network connectivity between services

#### Index Issues
- Check file permissions on knowledge documents
- Verify index file integrity: `ls -la chunks.jsonl vecs_n.npy`
- Rebuild if index is corrupted: `python cli.py build source.md`

#### Performance Issues
- Monitor memory usage during embedding
- Check if FAISS is properly configured
- Verify timeout values are appropriate

### Diagnostic Commands
```bash
# Check configuration
python -c "import clockify_rag.config as c; print(f'URL: {c.OLLAMA_URL}, Model: {c.GEN_MODEL}')"

# Test connectivity
python -c "from clockify_rag.error_handlers import check_endpoint_health; print(check_endpoint_health())"

# Run diagnostics
python clockify_support_cli_final.py chat --debug
```

## Backup and Recovery

### Index Backup
```bash
# Backup index files
tar -czf index_backup_$(date +%Y%m%d).tar.gz \
  chunks.jsonl vecs_n.npy bm25.json index.meta.json
```

### Configuration Backup
```bash
# Backup configuration
cp .env /backup/location/.env.backup
cp -r config/ /backup/location/config.backup/
```

## Scaling Considerations

### Horizontal Scaling
- Deploy multiple instances behind load balancer
- Use shared storage for index files
- Implement distributed caching if needed

### Vertical Scaling
- Increase embedding workers for indexing
- More memory for larger caches
- Faster storage for index I/O

This RAG system is designed for robust production deployment with proper error handling, monitoring, and operational concerns addressed.