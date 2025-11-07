# Quick Setup: Remote Company Ollama

**Target**: Connect RAG tool to `http://10.127.0.192:11434` (VPN required)
**Time**: 5 minutes
**Status**: ✅ **NO CODE CHANGES NEEDED**

---

## Prerequisites

1. ✅ VPN connected to company network
2. ✅ Python 3.7+ installed
3. ✅ Virtual environment activated (`source rag_env/bin/activate`)

---

## Step-by-Step Setup

### 1. Set Environment Variable (30 seconds)

```bash
# Temporary (this session only):
export OLLAMA_URL="http://10.127.0.192:11434"

# Or permanent (add to ~/.zshrc or ~/.bashrc):
echo 'export OLLAMA_URL="http://10.127.0.192:11434"' >> ~/.zshrc
source ~/.zshrc
```

### 2. Verify Connectivity (1 minute)

```bash
# Test version endpoint (must be on VPN!)
curl http://10.127.0.192:11434/api/version

# Expected output:
# {"version":"0.x.x"}

# List available models (verify qwen2.5:32b and nomic-embed-text)
curl http://10.127.0.192:11434/api/tags | python3 -m json.tool | grep name
```

**If this fails**: Check VPN connection

### 3. Optional: Adjust Timeouts for Network Latency (30 seconds)

```bash
# Add to ~/.zshrc for persistence
export EMB_READ_TIMEOUT=120        # 60→120s (remote embedding)
export CHAT_READ_TIMEOUT=180       # 120→180s (remote LLM)
export EMB_MAX_WORKERS=4           # 8→4 (avoid network saturation)
```

### 4. Build Knowledge Base (2-3 minutes)

```bash
# Recommended: Use local embeddings for faster build
EMB_BACKEND=local python3 clockify_support_cli_final.py build knowledge_full.md

# Or use remote Ollama embeddings (10-15 minutes):
# python3 clockify_support_cli_final.py build knowledge_full.md
```

### 5. Start Interactive Chat (instant)

```bash
python3 clockify_support_cli_final.py chat

# Try these test questions:
> How do I track time in Clockify?
> What are the pricing plans?
> :exit
```

---

## Complete One-Liner (Copy-Paste)

```bash
# Setup + Build + Chat
export OLLAMA_URL="http://10.127.0.192:11434" && \
export EMB_READ_TIMEOUT=120 && \
export CHAT_READ_TIMEOUT=180 && \
curl -s http://10.127.0.192:11434/api/version && \
echo "✅ Connected to company Ollama" && \
EMB_BACKEND=local python3 clockify_support_cli_final.py build knowledge_full.md && \
python3 clockify_support_cli_final.py chat
```

---

## Troubleshooting

### Error: "Connection refused" or "Connection timeout"
**Cause**: VPN not connected or Ollama service down
**Fix**:
1. Connect to company VPN
2. Test with: `curl http://10.127.0.192:11434/api/version`
3. Contact IT if Ollama service is down

### Error: "Model 'qwen2.5:32b' not found"
**Cause**: Model not pulled on server
**Fix**: Contact server admin to pull model:
```bash
# On Ollama server:
ollama pull qwen2.5:32b
ollama pull nomic-embed-text
```

### Error: "Read timeout" during build
**Cause**: Network latency or slow embeddings
**Fix**: Increase timeout:
```bash
export EMB_READ_TIMEOUT=300  # 5 minutes
```

### Build takes too long (>15 minutes)
**Solution**: Use local embeddings instead:
```bash
# Install SentenceTransformer (one-time):
pip install sentence-transformers

# Build with local embeddings (much faster):
EMB_BACKEND=local python3 clockify_support_cli_final.py build knowledge_full.md
```

---

## Configuration Cheat Sheet

### Required (Minimum Setup):
```bash
export OLLAMA_URL="http://10.127.0.192:11434"
```

### Recommended (Production):
```bash
export OLLAMA_URL="http://10.127.0.192:11434"
export EMB_READ_TIMEOUT=120        # Increase for network
export CHAT_READ_TIMEOUT=180       # Increase for large models
export EMB_MAX_WORKERS=4           # Reduce for remote endpoint
```

### Optional (Advanced):
```bash
export CACHE_MAXSIZE=200           # Larger cache (default: 100)
export CACHE_TTL=7200              # 2-hour cache (default: 3600)
export RATE_LIMIT_REQUESTS=20      # Queries/minute (default: 10)
export LOG_QUERY_INCLUDE_ANSWER=0  # Redact answers in logs (privacy)
export STRICT_CITATIONS=1          # Require citations (quality)
```

---

## Verify Setup

### Quick Health Check:
```bash
# 1. Environment variable set?
echo $OLLAMA_URL
# Expected: http://10.127.0.192:11434

# 2. Ollama accessible?
curl -s http://10.127.0.192:11434/api/version | grep version
# Expected: "version":"0.x.x"

# 3. Models available?
curl -s http://10.127.0.192:11434/api/tags | grep -E "(qwen2.5:32b|nomic-embed-text)"
# Expected: Two lines with model names

# 4. Knowledge base built?
ls -lh chunks.jsonl vecs_n.npy bm25.json
# Expected: Files exist with non-zero size

# 5. RAG working?
echo "How do I track time?" | python3 clockify_support_cli_final.py ask
# Expected: Answer with citations
```

---

## Performance Tips

### 1. Build Index Locally (5-10x faster):
```bash
# Use local SentenceTransformer for build
EMB_BACKEND=local python3 clockify_support_cli_final.py build knowledge_full.md

# Then use remote Ollama for queries (fast!)
python3 clockify_support_cli_final.py chat
```

### 2. Enable Query Caching (automatic, but verify):
```bash
# Default: 100 queries cached for 1 hour
# Increase for better performance:
export CACHE_MAXSIZE=200
export CACHE_TTL=7200
```

### 3. Monitor Cache Hit Rate:
```bash
# In Python REPL:
from clockify_rag.caching import get_query_cache
cache = get_query_cache()
print(cache.stats())
# {'hits': 45, 'misses': 55, 'size': 55, 'hit_rate': 0.45}
```

---

## What's Different from Local Ollama?

| Aspect | Local (127.0.0.1) | Remote (10.127.0.192) |
|--------|-------------------|----------------------|
| **Setup** | `ollama serve` | Set `OLLAMA_URL` env var |
| **Latency** | 50-100ms | 100-200ms (+50-100ms network) |
| **VPN Required** | No | Yes (always) |
| **Timeouts** | Default OK | Increase recommended |
| **Build Speed** | Fast | Slower (use local embeddings) |
| **Query Speed** | Fast | Slightly slower (cache helps) |
| **Reliability** | Self-managed | Shared service |

---

## Summary

✅ **Your codebase already supports remote Ollama perfectly!**

**Required steps**:
1. Set `OLLAMA_URL` environment variable
2. Verify connectivity with `curl`
3. Build knowledge base (use local embeddings for speed)
4. Start chatting!

**No code changes needed!** 🎉

---

## Next Steps

After setup works:
1. Read full audit: `COMPREHENSIVE_AUDIT_2025-11-07.md`
2. Review architecture: `CLAUDE.md`
3. Check thread safety: `pytest tests/test_thread_safety.py -v`
4. Run benchmarks: `python3 benchmark.py --quick`
5. Deploy to production (see deployment checklist in audit)

---

## Support

- **Full documentation**: See `COMPREHENSIVE_AUDIT_2025-11-07.md`
- **Remote Ollama analysis**: See `REMOTE_OLLAMA_ANALYSIS.md`
- **Architecture guide**: See `CLAUDE.md`
- **Quick start**: See `QUICKSTART.md`

**Created**: 2025-11-07 by comprehensive codebase audit
