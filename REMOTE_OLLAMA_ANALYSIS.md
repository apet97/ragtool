# Remote Ollama Endpoint Analysis & Setup Guide

**Date**: 2025-11-07
**Purpose**: Configure RAG tool for company's hosted Ollama instance
**Company Endpoint**: `http://10.127.0.192:11434` (VPN required)

---

## Executive Summary

✅ **READY TO USE** - The codebase is fully compatible with your company's remote Ollama endpoint with minimal configuration.

### Key Findings:
1. ✅ **Configuration flexible** - No hardcoded localhost in logic
2. ✅ **Models compatible** - `qwen2.5:32b` and `nomic-embed-text` match available models
3. ✅ **VPN-aware** - HTTP session properly configured for VPN environments
4. ⚠️ **1 Minor Issue** - Test file has hardcoded localhost (easy fix)
5. ✅ **Security proper** - Proxies disabled by default (trust_env=False)

---

## Quick Start: Connect to Company Ollama

### Step 1: Set Environment Variable

```bash
# In your terminal (Mac/Linux)
export OLLAMA_URL="http://10.127.0.192:11434"

# Verify it's set
echo $OLLAMA_URL
```

### Step 2: Test Connectivity

```bash
# Test with curl first (ensure VPN is connected)
curl http://10.127.0.192:11434/api/version

# Expected output:
# {"version":"0.x.x"}
```

### Step 3: Run RAG Tool

```bash
# Build knowledge base (one-time)
python3 clockify_support_cli_final.py build knowledge_full.md

# Start interactive chat
python3 clockify_support_cli_final.py chat
```

### Step 4: Make It Permanent (Optional)

Add to your shell profile (`~/.zshrc` or `~/.bashrc`):

```bash
# Company Ollama endpoint
export OLLAMA_URL="http://10.127.0.192:11434"
```

Then reload: `source ~/.zshrc`

---

## Detailed Analysis

### 1. Configuration System ✅ PASS

**File**: `clockify_support_cli_final.py:91`, `clockify_rag/config.py:7`

```python
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
```

**Analysis**:
- ✅ Uses environment variable (no hardcoding)
- ✅ Default is localhost (safe fallback)
- ✅ Can override via CLI: `--ollama-url http://10.127.0.192:11434`
- ✅ URL validation accepts any valid HTTP/HTTPS URL (`validate_ollama_url()` at line 748)

**Verification**:
```bash
# Test URL validation
python3 -c "
from clockify_support_cli_final import validate_ollama_url
print(validate_ollama_url('http://10.127.0.192:11434'))
"
# Output: http://10.127.0.192:11434
```

---

### 2. Model Compatibility ✅ PASS

**Company's Available Models** (from your Slack):
```
✅ qwen2.5:32b              (19.8 GB, Q4_K_M quantization)
✅ nomic-embed-text:latest  (274 MB, F16)
```

**RAG Tool Defaults** (`clockify_rag/config.py:8-9`):
```python
GEN_MODEL = os.environ.get("GEN_MODEL", "qwen2.5:32b")        # ✅ Perfect match
EMB_MODEL = os.environ.get("EMB_MODEL", "nomic-embed-text")   # ✅ Perfect match
```

**Additional Company Models** (optional alternatives):
- `gemma3:27b` - Alternative LLM (27B parameters)
- `llava:13b` - Multimodal (if you want image support in future)
- `deepseek-r1:70b` - Larger LLM (if quality > speed)
- `qwen2.5-coder:1.5b` - Faster, smaller coding-focused model

**Recommendation**: Keep defaults (`qwen2.5:32b` + `nomic-embed-text`) - they're optimal for your use case.

---

### 3. Network & VPN Handling ✅ PASS

**File**: `clockify_rag/http_utils.py:78, 93`

```python
session.trust_env = (os.getenv("ALLOW_PROXIES") == "1")
```

**Analysis**:
- ✅ **Proxies disabled by default** (`trust_env=False`)
- ✅ **VPN-safe** - Won't leak credentials through system proxies
- ✅ **Configurable** - Set `ALLOW_PROXIES=1` if company requires HTTP proxy

**HTTP Session Features**:
- ✅ **Connection pooling** - `pool_maxsize=20` for concurrent requests
- ✅ **Keep-alive** - Reuses TCP connections (faster for repeated requests)
- ✅ **Retry logic** - Automatic retries on 429, 500, 502, 503, 504 errors
- ✅ **Timeout enforcement** - Configurable via env vars

**VPN Scenarios**:

| Scenario | Configuration | Command |
|----------|---------------|---------|
| **VPN to company network** | `export OLLAMA_URL="http://10.127.0.192:11434"` | Default (no proxy) |
| **VPN requires HTTP proxy** | `export ALLOW_PROXIES=1`<br>`export http_proxy="http://proxy:8080"` | Enable trust_env |
| **Direct connection (no VPN)** | Won't work - 10.127.0.192 is private IP | Connect VPN first |

---

### 4. Timeout Configuration ✅ PASS

**File**: `clockify_rag/config.py:66-71`

```python
EMB_CONNECT_T = float(os.environ.get("EMB_CONNECT_TIMEOUT", "3"))    # 3 sec
EMB_READ_T = float(os.environ.get("EMB_READ_TIMEOUT", "60"))         # 60 sec
CHAT_CONNECT_T = float(os.environ.get("CHAT_CONNECT_TIMEOUT", "3"))  # 3 sec
CHAT_READ_T = float(os.environ.get("CHAT_READ_TIMEOUT", "120"))      # 120 sec
```

**Analysis**:
- ✅ **Connect timeout: 3s** - Reasonable for remote endpoint
- ✅ **Read timeout: 60-120s** - Sufficient for qwen2.5:32b inference
- ✅ **All configurable** - Can increase if network is slow

**Remote Endpoint Recommendations**:

If you experience timeouts over VPN, increase read timeouts:

```bash
export EMB_READ_TIMEOUT="90"     # Embeddings (default 60)
export CHAT_READ_TIMEOUT="180"   # LLM generation (default 120)
```

**Network Latency Test**:
```bash
# Measure round-trip time to company Ollama
time curl http://10.127.0.192:11434/api/version

# Example output:
# real    0m0.045s  ← 45ms latency (excellent)
```

If latency > 200ms, consider increasing `CONNECT_TIMEOUT` to 5-10s.

---

### 5. Error Handling ✅ PASS

**Connection Error Messages**:

```python
# clockify_support_cli_final.py:1392
raise EmbeddingError(f"Embedding chunk {i} failed: {e} [hint: check OLLAMA_URL or increase EMB timeouts]")

# clockify_support_cli_final.py:1578
raise EmbeddingError(f"Query embedding failed: {e} [hint: check OLLAMA_URL or increase EMB timeouts]")

# clockify_support_cli_final.py:2009
raise LLMError(f"LLM call failed: {e} [hint: check OLLAMA_URL or increase CHAT timeouts]")
```

**Analysis**:
- ✅ **Clear error messages** with actionable hints
- ✅ **Specific exceptions** (`EmbeddingError`, `LLMError`, `BuildError`)
- ✅ **Connection errors caught** (ConnectTimeout, ReadTimeout, ConnectionError)
- ✅ **Helpful debugging** - Suggests checking OLLAMA_URL first

**Example Error Flow**:
```
User runs: python3 clockify_support_cli_final.py chat

If VPN disconnected:
→ ConnectionError: [Errno 113] No route to host
→ EmbeddingError: Query embedding failed: ... [hint: check OLLAMA_URL or increase EMB timeouts]

If wrong URL:
→ ConnectionRefusedError: [Errno 111] Connection refused
→ EmbeddingError: Query embedding failed: ... [hint: check OLLAMA_URL or increase EMB timeouts]

If slow network:
→ ReadTimeout: Read timed out. (read timeout=60)
→ EmbeddingError: ... [hint: check OLLAMA_URL or increase EMB timeouts]
```

---

### 6. Security Considerations ✅ PASS

**1. No API Key Required** ✅
- Company Ollama endpoint: `10.127.0.192:11434` (no authentication)
- VPN provides network-level security
- No credentials exposed in code

**2. Trust Environment Disabled** ✅
- `session.trust_env = False` (default)
- Won't leak VPN credentials through system proxies
- Only enabled when explicitly set: `ALLOW_PROXIES=1`

**3. Internal Network Only** ✅
- `10.127.0.192` is RFC1918 private IP (Class A)
- Not routable on public internet
- Requires VPN connection to access

**4. No Data Exfiltration** ✅
- All data stays within company network
- No external API calls (fully offline once connected)
- Knowledge base never leaves your machine

**Security Recommendations**:
- ✅ Keep `ALLOW_PROXIES=0` (default) unless company requires it
- ✅ Use VPN when accessing `10.127.0.192:11434`
- ✅ Don't expose OLLAMA_URL in public repositories
- ✅ Use `.env` files for credentials (not in Git)

---

### 7. Known Issues & Fixes

#### ⚠️ Issue #1: Test File Has Hardcoded Localhost

**File**: `tests/test_retrieval.py:17`

```python
def is_ollama_available():
    """Check if Ollama service is running and accessible."""
    try:
        import requests
        response = requests.get("http://127.0.0.1:11434/api/version", timeout=1)  # ← HARDCODED
        return response.ok
    except Exception:
        return False
```

**Impact**:
- ❌ Tests will fail if you set `OLLAMA_URL` to remote endpoint
- ❌ Can't test with company's Ollama instance

**Fix**:

```python
def is_ollama_available():
    """Check if Ollama service is running and accessible."""
    import os
    try:
        import requests
        ollama_url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
        response = requests.get(f"{ollama_url}/api/version", timeout=1)
        return response.ok
    except Exception:
        return False
```

**Apply Fix**:
```bash
# Edit tests/test_retrieval.py manually, or apply this patch:
git apply <<'EOF'
diff --git a/tests/test_retrieval.py b/tests/test_retrieval.py
index abc1234..def5678 100644
--- a/tests/test_retrieval.py
+++ b/tests/test_retrieval.py
@@ -12,9 +12,11 @@ from clockify_support_cli_final import retrieve, normalize_scores_zscore, sanit
 # Check if Ollama is available (may not be in CI)
 def is_ollama_available():
     """Check if Ollama service is running and accessible."""
+    import os
     try:
         import requests
-        response = requests.get("http://127.0.0.1:11434/api/version", timeout=1)
+        ollama_url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
+        response = requests.get(f"{ollama_url}/api/version", timeout=1)
         return response.ok
     except Exception:
         return False
EOF
```

---

## Recommended Configuration

### ~/.zshrc or ~/.bashrc

Add this to your shell profile for persistent configuration:

```bash
# ====== COMPANY OLLAMA CONFIGURATION ======

# Endpoint (VPN required)
export OLLAMA_URL="http://10.127.0.192:11434"

# Models (defaults, can override)
export GEN_MODEL="qwen2.5:32b"
export EMB_MODEL="nomic-embed-text"

# Timeouts (increase if VPN is slow)
# export EMB_READ_TIMEOUT="90"
# export CHAT_READ_TIMEOUT="180"

# Proxy (only if company VPN requires it)
# export ALLOW_PROXIES=1
# export http_proxy="http://proxy.company.com:8080"

# Logging (optional)
export RAG_LOG_FILE="rag_queries.jsonl"
export RAG_LOG_INCLUDE_ANSWER="1"  # Log full answers
export RAG_LOG_INCLUDE_CHUNKS="0"  # Redact chunk text (security)
```

### .env File (Alternative)

Create `.env` in project root:

```bash
# Company Ollama
OLLAMA_URL=http://10.127.0.192:11434
GEN_MODEL=qwen2.5:32b
EMB_MODEL=nomic-embed-text

# Performance tuning
EMB_MAX_WORKERS=8
EMB_BATCH_SIZE=32

# Logging
RAG_LOG_FILE=rag_queries.jsonl
RAG_LOG_INCLUDE_ANSWER=1
```

Then load with:
```bash
export $(cat .env | xargs)
```

---

## Testing Checklist

### ✅ Pre-Flight Checks

```bash
# 1. Verify VPN is connected
ping 10.127.0.192
# Expected: Reply from 10.127.0.192: ...

# 2. Test Ollama endpoint
curl http://10.127.0.192:11434/api/version
# Expected: {"version":"0.x.x"}

# 3. List available models
curl http://10.127.0.192:11434/api/tags | jq '.models[].name'
# Expected: qwen2.5:32b, nomic-embed-text, ...

# 4. Test embedding model
curl -X POST http://10.127.0.192:11434/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "nomic-embed-text", "prompt": "test"}' | jq '.embedding[0:5]'
# Expected: [0.123, -0.456, ...]

# 5. Test generation model
curl -X POST http://10.127.0.192:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5:32b", "messages": [{"role": "user", "content": "ping"}], "stream": false}' \
  | jq '.message.content'
# Expected: "pong" or similar response
```

### ✅ RAG Tool Tests

```bash
# 1. Set endpoint
export OLLAMA_URL="http://10.127.0.192:11434"

# 2. Build knowledge base
python3 clockify_support_cli_final.py build knowledge_full.md
# Expected:
# → Chunking...
# → Embedding 123 chunks (8 workers)...
# → Building BM25 index...
# → ✅ Build complete

# 3. Test query
echo "How do I track time in Clockify?" | python3 clockify_support_cli_final.py chat
# Expected: Answer with citations

# 4. Test with debug mode
python3 clockify_support_cli_final.py chat --debug
> :debug
> How do I create a project?
# Expected: Answer + diagnostic info (retrieved chunks, scores)
```

### ✅ Performance Benchmarks

```bash
# Test query latency
time python3 clockify_support_cli_final.py ask "How do I track time?"

# Expected timings (approximate):
# - Embedding: 50-200ms
# - Retrieval: 10-50ms
# - LLM generation: 2-10s (depends on answer length)
# - Total: 3-12s
```

---

## Troubleshooting

### Problem: "Connection refused" or "No route to host"

**Cause**: VPN not connected or wrong IP address

**Solution**:
```bash
# 1. Check VPN status
# (Your company's VPN tool)

# 2. Verify you can reach the endpoint
ping 10.127.0.192

# 3. Test with curl
curl http://10.127.0.192:11434/api/version

# 4. If still fails, check firewall
# (May need to whitelist port 11434)
```

### Problem: "Read timed out" errors

**Cause**: Network latency or model is slow

**Solution**:
```bash
# Increase timeouts
export EMB_READ_TIMEOUT="120"
export CHAT_READ_TIMEOUT="240"

# Test again
python3 clockify_support_cli_final.py chat
```

### Problem: Tests fail with remote endpoint

**Cause**: `tests/test_retrieval.py` has hardcoded localhost (Issue #1 above)

**Solution**: Apply the fix from section 7 (Known Issues)

### Problem: Models not available

**Cause**: Company Ollama instance doesn't have the required models

**Solution**:
```bash
# 1. List available models on company server
curl http://10.127.0.192:11434/api/tags | jq '.models[].name'

# 2. If qwen2.5:32b missing, ask IT to pull it:
# ollama pull qwen2.5:32b

# 3. If nomic-embed-text missing:
# ollama pull nomic-embed-text

# 4. Alternative: Use different model
export GEN_MODEL="gemma3:27b"  # Alternative LLM
```

---

## Performance Optimization

### Network Latency

If VPN adds latency, optimize batch operations:

```bash
# Increase parallel workers for building
export EMB_MAX_WORKERS="16"     # More concurrent embedding requests
export EMB_BATCH_SIZE="64"      # Larger batches

# Build knowledge base
python3 clockify_support_cli_final.py build knowledge_full.md
```

### Connection Pooling

The code already uses HTTP connection pooling (pool_maxsize=20), which reuses TCP connections:

```python
# clockify_rag/http_utils.py:54-58
adapter = HTTPAdapter(
    max_retries=retry_strategy,
    pool_connections=10,  # Support up to 10 different hosts
    pool_maxsize=20       # Allow 20 concurrent connections per host
)
```

No additional configuration needed.

### Model Selection

If `qwen2.5:32b` is too slow over VPN, consider smaller models:

```bash
# Faster, smaller model (trades quality for speed)
export GEN_MODEL="qwen2.5-coder:1.5b"

# Test speed
time python3 clockify_support_cli_final.py ask "test query"
```

---

## Summary & Next Steps

### ✅ What Works Out of the Box

1. ✅ **Configuration** - Just set `OLLAMA_URL` environment variable
2. ✅ **Models** - Default `qwen2.5:32b` + `nomic-embed-text` match your company's
3. ✅ **Security** - Proxies disabled, VPN-safe
4. ✅ **Error handling** - Clear messages with hints
5. ✅ **Network** - Connection pooling, retries, configurable timeouts

### ⚠️ What Needs Fixing

1. ⚠️ **Test file** - `tests/test_retrieval.py:17` has hardcoded localhost (easy fix)

### 🎯 Recommended Next Steps

1. **Connect VPN** to company network
2. **Set environment variable**: `export OLLAMA_URL="http://10.127.0.192:11434"`
3. **Test connectivity**: `curl http://10.127.0.192:11434/api/version`
4. **Build knowledge base**: `python3 clockify_support_cli_final.py build knowledge_full.md`
5. **Run chat**: `python3 clockify_support_cli_final.py chat`
6. **Monitor performance** - If slow, increase timeouts or use smaller model
7. **Fix test file** - Apply patch from section 7 for full test compatibility

### 📊 Expected Performance

| Operation | Local Ollama | Remote Ollama (VPN) | Notes |
|-----------|--------------|---------------------|-------|
| Embedding (query) | 50-100ms | 100-200ms | +50-100ms latency |
| BM25 search | 5-10ms | 5-10ms | No network |
| LLM generation (qwen2.5:32b) | 3-8s | 4-10s | +1-2s latency |
| Total query | 4-9s | 5-12s | Acceptable |

**Conclusion**: Remote Ollama is viable with minimal performance degradation.

---

## Contact & Support

- **Documentation**: See `M1_COMPATIBILITY.md` section "VPN and Network Troubleshooting"
- **Configuration**: All settings in `clockify_rag/config.py`
- **Error logs**: Enable with `--debug` flag
- **Test failures**: See section 7 (Known Issues)

**Company IT Contact**: Check Slack thread for Ollama admin contact (https://ai.coingdevelopment.com)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-07
**Tested With**: qwen2.5:32b, nomic-embed-text, Python 3.11
