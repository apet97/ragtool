# Smoke Test & Pre-Deploy Runbook

This runbook provides step-by-step verification procedures for the RAG system before deployment or after changes.

## Quick Smoke Test (2-5 minutes)

For rapid validation after code changes or before deployment:

```bash
# 1. Verify environment
python scripts/verify_env.py

# 2. Verify import fallback behavior
python scripts/verify_fallback.py

# 3. Run Python smoke test (uses mock LLM, no VPN needed)
python scripts/smoke_rag.py

# 4. Run sanity check (requires VPN/Ollama)
python -m clockify_rag.sanity_check
```

**Expected Results:**
- All checks show ✅ green checkmarks
- verify_fallback.py: 6/6 scenarios passed
- Python smoke test returns exit code 0
- Sanity check passes all 5 checks (or 4/5 if VPN down)

**If any fail:**
- Check error messages for specific issues
- See "Troubleshooting" section below

---

## Full Smoke Test Suite (5-10 minutes)

For comprehensive validation including integration with real Ollama:

```bash
# Run full smoke test suite (requires VPN connection)
./scripts/smoke.sh
```

This runs:
1. Virtual environment activation
2. Dependency checks
3. Knowledge base build
4. Self-test
5. JSON query test
6. Plain text query test
7. KPI log verification

**Expected Results:**
- All 7 steps complete with ✅
- Log file `smoke.log` created with detailed output
- No ❌ error markers

---

## Pre-Deployment Checklist

Before deploying to production or CI, verify:

### 1. Dependencies

```bash
# Check Python version (need 3.8+)
python --version

# Verify langchain-ollama is installed (REQUIRED for production)
python -c "import langchain_ollama; print('✅ langchain-ollama installed')"

# Set production environment
export ENVIRONMENT=production
```

### 2. Configuration

```bash
# Verify critical environment variables
echo "ENVIRONMENT: ${ENVIRONMENT:-not set}"
echo "RAG_OLLAMA_URL: ${RAG_OLLAMA_URL:-not set}"
echo "DEFAULT_TOP_K: ${DEFAULT_TOP_K:-using default 15}"
echo "MAX_TOP_K: ${MAX_TOP_K:-using default 50}"
```

### 3. Index Artifacts

```bash
# Check required files exist
ls -lh chunks.jsonl vecs_n.npy meta.jsonl bm25.json

# Verify index is not stale
python -c "
import json
from pathlib import Path
if Path('index.meta.json').exists():
    with open('index.meta.json') as f:
        meta = json.load(f)
    print(f\"Index built: {meta.get('build_timestamp')}\")
    print(f\"Knowledge MD5: {meta.get('knowledge_md5')}\")
else:
    print('⚠️  index.meta.json not found')
"
```

### 4. Connectivity

```bash
# Verify Ollama endpoint is reachable
curl -s "${RAG_OLLAMA_URL:-http://10.127.0.192:11434}/api/tags" | head -20

# Or use Python helper
python -c "
from clockify_rag.utils import check_ollama_connectivity
from clockify_rag import config
connected, message = check_ollama_connectivity()
print(f\"{'✅' if connected else '❌'} {message}\")
"
```

### 5. Unit Tests

```bash
# Run tests that don't require VPN
pytest -m "not integration" -v

# If VPN is connected, run integration tests too
pytest -m integration -v
```

### 6. End-to-End Test

```bash
# Single query test with real LLM (requires VPN)
python scripts/smoke_rag.py --client ollama --question "How do I track time in Clockify?"

# Expected: Non-empty answer, confidence > 0, no LLM errors
```

---

## Troubleshooting

### Issue: `langchain-ollama not found` in Production

**Symptom:**
```
ImportError: langchain-ollama is required in production but not installed.
```

**Fix:**
```bash
pip install langchain-ollama
# Or upgrade if already installed
pip install --upgrade langchain-ollama
```

### Issue: Ollama Timeout or Connection Error

**Symptom:**
```
❌ Ollama timeout at http://10.127.0.192:11434 (VPN down?)
```

**Fix:**
1. Check VPN connection: `ping 10.127.0.192`
2. Verify Ollama is running: `curl http://10.127.0.192:11434/api/tags`
3. Check firewall settings
4. Increase timeout: `export OLLAMA_TIMEOUT=300`

### Issue: Missing Index Artifacts

**Symptom:**
```
❌ Index artifacts not found. Run `make build` or `ragctl ingest` first.
```

**Fix:**
```bash
# Build index from knowledge base
make build
# Or
python clockify_support_cli_final.py build knowledge_full.md
```

### Issue: Test Failures After Update

**Symptom:**
```
FAILED tests/test_retrieval.py::test_top_k_default
```

**Fix:**
1. Check if config changed: `git diff config.py`
2. Re-run tests in isolation: `pytest tests/test_retrieval.py::test_top_k_default -v`
3. Clear pytest cache: `rm -rf .pytest_cache`
4. Reinstall in editable mode: `pip install -e .[dev]`

### Issue: Context Overflow Warnings

**Symptom:**
```
WARNING: top_k=100 exceeds MAX_TOP_K=50, clamping to MAX_TOP_K
```

**Fix:**
```bash
# Either reduce top_k in config
export DEFAULT_TOP_K=15

# Or increase MAX_TOP_K if you have a large-context model
export MAX_TOP_K=100
```

---

## Quick Reference: Common Commands

```bash
# Environment check
python scripts/verify_env.py

# Fallback verification (unit test, no VPN needed)
python scripts/verify_fallback.py

# Sanity check (requires VPN)
python -m clockify_rag.sanity_check

# Smoke test (offline, uses mock LLM)
python scripts/smoke_rag.py

# Smoke test (online, real LLM)
python scripts/smoke_rag.py --client ollama

# Full smoke suite (requires VPN)
./scripts/smoke.sh

# Unit tests only (no VPN needed)
pytest -m "not integration"

# Integration tests (needs VPN)
pytest -m integration

# All tests
pytest

# Build index
make build

# Interactive CLI
make chat
```

---

## Environment Variables Summary

```bash
# Critical for production
export ENVIRONMENT=production           # Enforce strict dependency checks
export RAG_OLLAMA_URL=http://your-ollama:11434

# Retrieval tuning
export DEFAULT_TOP_K=15                 # Default retrieval fan-out
export MAX_TOP_K=50                     # Hard ceiling to prevent overflow
export DEFAULT_PACK_TOP=8               # Snippets in context
export CTX_TOKEN_BUDGET=12000           # Token budget for context

# Performance
export OLLAMA_TIMEOUT=120               # Timeout for Ollama calls
export DEFAULT_RETRIES=2                # Retry count for LLM
export EMB_MAX_WORKERS=8                # Parallel embedding workers

# Optional
export CHUNK_CHARS=1600                 # Chunk size
export CHUNK_OVERLAP=200                # Overlap between chunks
export USE_INTENT_CLASSIFICATION=1      # Enable intent-based retrieval
```

---

## Success Criteria

Before marking a deployment as "ready":

- [ ] `python scripts/verify_env.py` passes all checks
- [ ] `python -m clockify_rag.sanity_check` exits with code 0
- [ ] `pytest -m "not integration"` all pass
- [ ] `pytest -m integration` all pass (with VPN)
- [ ] `python scripts/smoke_rag.py --client ollama` returns valid answer
- [ ] ENVIRONMENT is set to "production" or "ci"
- [ ] langchain-ollama is installed (not relying on fallback)
- [ ] Index artifacts exist and are not stale
- [ ] Ollama endpoint is reachable

---

## Next Steps After Validation

1. **Tag the release**:
   ```bash
   git tag -a v5.x.x -m "Release v5.x.x - [brief description]"
   git push origin v5.x.x
   ```

2. **Deploy to staging/production**:
   - Follow deployment procedures in `PRODUCTION_GUIDE.md`
   - Monitor logs for first 10-20 queries
   - Verify metrics endpoints are working

3. **Post-deployment verification**:
   ```bash
   # Test deployed endpoint
   curl http://your-deployment:8000/health
   curl http://your-deployment:8000/metrics

   # Test query
   curl -X POST http://your-deployment:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "How do I track time?"}'
   ```
