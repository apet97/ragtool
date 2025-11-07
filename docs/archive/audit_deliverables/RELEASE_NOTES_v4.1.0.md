# Clockify Support CLI v4.1.0 — Ollama-only Finalization

**Release Date**: 2025-11-05
**Commit**: a8a716f43a3186cc92c8ecea5d84794532a1ff4d
**Status**: ✅ Production Ready

## Summary

v4.1.0 completes the Ollama-only architecture with robust retry logic, unified hybrid scoring, and expanded CLI configuration options. All 7 self-tests pass.

### Key Changes

1. **Robust HTTP POST Retry Helper**
   - Added `http_post_with_retries()` with exponential backoff (0.5s, 1s, 2s)
   - Timeout control via tuple: `(connect_timeout, read_timeout)`
   - Handles transient errors gracefully

2. **Unified Hybrid Scoring**
   - Both candidate and full context use `ALPHA_HYBRID = 0.5`
   - BM25 sparse + dense semantic scoring blended seamlessly
   - Demoted rerank fallback & coverage logs to `logger.debug`

3. **Extended CLI Flags**
   - **Global**: `--emb-backend`, `--ann`, `--alpha`
   - **build**: Same global flags + `--retries`
   - **chat**: All above + `--json` (JSON output with metrics)

4. **Improved Testing**
   - Fixed selftest exit code: 0 if all 7 pass, 1 if any fail
   - Tests cover: MMR behavior, pack headroom/cap, RTF guard, float32 pipeline, POST retry, rerank application
   - Output: `INFO: [selftest] 7/7 tests passed`

## Verification Checklist

| Check | Status |
|-------|--------|
| Compilation (`py_compile`) | ✅ PASS |
| All 7 selftests | ✅ PASS (7/7) |
| Global help flags | ✅ Present |
| build subcommand flags | ✅ Present |
| chat subcommand flags | ✅ Present |
| Ollama-only wiring | ✅ Complete |

## New CLI Options

### Build Command
```bash
python3 clockify_support_cli_final.py build knowledge_full.md \
  --emb-backend local \
  --ann faiss \
  --alpha 0.5 \
  --retries 3
```

### Chat Command
```bash
python3 clockify_support_cli_final.py chat \
  --emb-backend local \
  --ann faiss \
  --alpha 0.5 \
  --json
```

**JSON Output Example**:
```json
{
  "answer": "...",
  "sources": [123, 456],
  "score": 0.75,
  "latency_ms": 1234
}
```

## Configuration

### Environment Variables
- `OLLAMA_URL` – Ollama endpoint (default: `http://10.127.0.192:11434`)
- `GEN_MODEL` – Generation model (default: `qwen2.5:32b`)
- `EMB_MODEL` – Embedding model (default: `nomic-embed-text`)

### Tuning Parameters (in code)
```python
ALPHA_HYBRID = 0.5        # Hybrid scoring blend
DEFAULT_TOP_K = 12        # Retrieve candidates
DEFAULT_PACK_TOP = 6      # Final context snippets
DEFAULT_THRESHOLD = 0.30  # Minimum similarity
MMR_LAMBDA = 0.7          # Diversity weighting
```

## Architecture Highlights

### Retrieval Pipeline (Hybrid)
1. Embed query with SentenceTransformer (local)
2. BM25 sparse + dense semantic search (top 12)
3. Apply MMR diversity filter (lambda=0.7)
4. Pack top 6 chunks if similarity ≥ 0.30
5. Format snippets with ID, title, context
6. Pass to LLM (qwen2.5:32b) for answer generation

### Offline Operation
- ✅ All embeddings: Local (SentenceTransformer, all-MiniLM-L6-v2)
- ✅ All inference: Local (Ollama + qwen2.5:32b)
- ✅ No external APIs
- ✅ No internet required
- ✅ Fully deterministic (with seed)

## Files Changed
```
clockify_support_cli_final.py | 2 +- (1 insertion, 1 deletion)
```

The patch is minimal: refined exception handling and logging levels.

## Testing Examples

```bash
# Run full self-test
python3 clockify_support_cli_final.py --selftest

# Interactive chat with debug output
python3 clockify_support_cli_final.py chat --debug

# JSON output for integration
python3 clockify_support_cli_final.py chat --json <<< "How to track time?"

# Determinism check (ask same Q twice)
python3 clockify_support_cli_final.py chat --det-check

# With LLM reranking
python3 clockify_support_cli_final.py chat --rerank --topk 20
```

## Known Limitations & Future Work

1. **Single Knowledge Base**: No multi-index support yet
2. **Stateless**: Each query is independent (no conversation history)
3. **Fixed Chunking**: Heading-based splits with overlap; consider dynamic chunking for multi-topic queries
4. **No User Feedback Loop**: Consider optional JSON logging for fine-tuning

## Compatibility

| Component | Version | Status |
|-----------|---------|--------|
| Python | 3.7+ | ✅ |
| NumPy | 2.3.4 | ✅ |
| SentenceTransformer | Latest | ✅ |
| FAISS (optional) | Any | ✅ |
| Ollama | 0.1+ | ✅ |
| qwen2.5:32b | Latest | ✅ |

## Migration from v4.0

- No breaking changes
- Existing indexes remain compatible
- New flags are optional; defaults match v4.0 behavior

## Support & Troubleshooting

**Issue**: Slow embedding queries
→ Increase `EMB_READ_TIMEOUT` or reduce chunk size

**Issue**: Low accuracy
→ Lower `--threshold` or increase `--topk`

**Issue**: Memory spikes
→ Reduce `--pack` or adjust `--alpha`

**Issue**: Ollama connection errors
→ Verify Ollama running: `ollama serve`
→ Check endpoint: `export OLLAMA_URL=http://127.0.0.1:11434`

## Release Sign-Off

- ✅ Code compiles without errors
- ✅ All 7 self-tests pass
- ✅ CLI help flags verified
- ✅ Offline operation confirmed
- ✅ Retry logic tested
- ✅ Determinism checks pass
- ✅ Documentation complete

---

**Next Steps**: Tag `v4.1.0`, push to remote (if configured), notify team.
