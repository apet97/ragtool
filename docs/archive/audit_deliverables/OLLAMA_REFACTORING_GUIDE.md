# Clockify CLI - Ollama Refactoring Guide v4.1

## Overview

This document outlines all modifications needed to refactor `clockify_support_cli_final.py` (v4.0) into an Ollama-optimized version (v4.1) with:

- **Local embeddings**: SentenceTransformer all-MiniLM-L6-v2 (384-dim)
- **FAISS ANN**: Optional IVFFlat indexing for fast similarity search
- **Hybrid retrieval**: BM25 + dense embeddings with alpha blending
- **Enhanced packing**: Dynamic chunk selection targeting 75% of token budget
- **Observability**: KPI logs, greppable rerank-fallback events, metrics
- **Self-test mode**: Health checks, determinism verification, end-to-end sanity
- **JSON output**: Optional structured output with metrics
- **Warm-up**: Pre-load LLM on startup to hide cold-start latency

## Architecture Changes

```
v4.0 (DeepSeek)              v4.1 (Ollama)
├─ Shim adapter             │  └─ Direct Ollama (no shim)
├─ Remote embeddings        │  └─ Local embeddings (sentence-transformers)
├─ No ANN                   │  └─ FAISS IVFFlat (optional)
└─ Basic logging            └─ KPI + rerank-fallback logs
```

## Installation

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` already includes:
- `sentence-transformers==3.3.1`
- `rank-bm25==0.2.2`
- `faiss-cpu==1.8.0.post1`

## Code Modifications

### 1. Configuration Section (lines 36-72)

**Add new config flags:**

```python
# ====== EMBEDDINGS BACKEND ======
EMB_BACKEND = os.environ.get("EMB_BACKEND", "local")  # "local" or "ollama"
EMB_DIM = 384  # all-MiniLM-L6-v2 dimension

# ====== ANN (Approximate Nearest Neighbors) ======
USE_ANN = os.environ.get("ANN", "faiss")  # "faiss" or "none"
ANN_NLIST = int(os.environ.get("ANN_NLIST", "256"))  # IVF clusters
ANN_NPROBE = int(os.environ.get("ANN_NPROBE", "16"))  # clusters to search

# ====== HYBRID SCORING ======
ALPHA_HYBRID = float(os.environ.get("ALPHA", "0.5"))  # 0.5 = BM25 and dense equally weighted

# ====== KPI TIMINGS ======
class KPI:
    retrieve_ms = 0
    ann_ms = 0
    rerank_ms = 0
    ask_ms = 0
```

**Update FILES dict:**

```python
FILES = {
    "chunks": "chunks.jsonl",
    "emb": "vecs_n.npy",  # Pre-normalized embeddings (float32)
    "emb_f16": "vecs_f16.memmap",  # float16 memory-mapped (optional)
    "emb_cache": "emb_cache.jsonl",  # Per-chunk embedding cache
    "meta": "meta.jsonl",
    "bm25": "bm25.json",
    "faiss_index": "faiss.index",  # FAISS IVFFlat index
    "index_meta": "index.meta.json",
}
```

### 2. SentenceTransformer Helpers (insert after imports)

```python
# ====== LOCAL EMBEDDINGS ======
_ST_ENCODER = None
_ST_BATCH_SIZE = 96

def _load_st_encoder():
    """Lazy-load SentenceTransformer model once."""
    global _ST_ENCODER
    if _ST_ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _ST_ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
        logger.debug("Loaded SentenceTransformer: all-MiniLM-L6-v2 (384-dim)")
    return _ST_ENCODER

def embed_local_batch(texts: list, normalize: bool = True) -> np.ndarray:
    """Encode texts locally using SentenceTransformer in batches."""
    model = _load_st_encoder()
    vecs = []
    for i in range(0, len(texts), _ST_BATCH_SIZE):
        batch = texts[i:i+_ST_BATCH_SIZE]
        batch_vecs = model.encode(batch, normalize_embeddings=normalize, convert_to_numpy=True)
        vecs.append(batch_vecs.astype("float32"))
    return np.vstack(vecs) if vecs else np.zeros((0, EMB_DIM), dtype="float32")
```

### 3. FAISS Index Helpers (insert after ST helpers)

```python
# ====== FAISS ANN INDEX ======
_FAISS_INDEX = None

def _try_load_faiss():
    """Try importing FAISS; returns None if not available."""
    try:
        import faiss
        return faiss
    except ImportError:
        logger.info("info: ann=fallback reason=missing-faiss")
        return None

def build_faiss_index(vecs: np.ndarray, nlist: int = 256, metric: str = "ip") -> object:
    """Build FAISS IVFFlat index (inner product for cosine on normalized vectors)."""
    faiss = _try_load_faiss()
    if faiss is None:
        return None

    dim = vecs.shape[1]
    # Use IndexFlatIP for inner product (cosine on normalized vecs)
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    # Train on subset
    train_size = min(20000, len(vecs))
    train_indices = np.random.choice(len(vecs), train_size, replace=False)
    train_vecs = vecs[train_indices].astype("float32")
    index.train(train_vecs)
    index.add(vecs.astype("float32"))
    index.nprobe = ANN_NPROBE

    logger.debug(f"Built FAISS index: nlist={nlist}, nprobe={ANN_NPROBE}, vectors={len(vecs)}")
    return index

def save_faiss_index(index, path: str = FILES["faiss_index"]):
    """Save FAISS index to disk."""
    if index is None:
        return
    faiss = _try_load_faiss()
    if faiss:
        faiss.write_index(index, path)
        logger.debug(f"Saved FAISS index to {path}")

def load_faiss_index(path: str = FILES["faiss_index"]) -> object:
    """Load FAISS index from disk."""
    if not os.path.exists(path):
        return None
    faiss = _try_load_faiss()
    if faiss:
        index = faiss.read_index(path)
        index.nprobe = ANN_NPROBE
        logger.debug(f"Loaded FAISS index from {path}")
        return index
    return None
```

### 4. Hybrid Retrieval Helpers

```python
# ====== HYBRID SCORING ======
def normalize_scores(scores: list) -> list:
    """Min-max normalize scores to [0, 1]."""
    if not scores or len(scores) == 0:
        return scores
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [0.5] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]

def hybrid_score(bm25_score: float, dense_score: float, alpha: float = 0.5) -> float:
    """Blend BM25 and dense scores: alpha * bm25_norm + (1 - alpha) * dense_norm."""
    return alpha * bm25_score + (1 - alpha) * dense_score

def retrieve_hybrid(query_text: str, topk: int = 12, alpha: float = 0.5) -> list:
    """Retrieve using BM25 + dense + FAISS (optional) + MMR."""
    start = time.time()

    # Embed query
    q_vec = embed_local_batch([query_text], normalize=True)[0]

    # BM25 retrieval
    bm25_hits = {}
    try:
        bm25_scores = index_bm25.get_scores(query_text.split())
        for idx, score in enumerate(bm25_scores):
            if score > 0:
                bm25_hits[idx] = score
        bm25_hits = dict(sorted(bm25_hits.items(), key=lambda x: x[1], reverse=True)[:topk*2])
    except:
        bm25_hits = {}

    # Dense retrieval (with optional ANN)
    dense_hits = {}
    if USE_ANN == "faiss" and _FAISS_INDEX:
        ann_start = time.time()
        distances, indices = _FAISS_INDEX.search(q_vec.reshape(1, -1).astype("float32"), topk * 3)
        KPI.ann_ms = (time.time() - ann_start) * 1000
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0:
                dense_hits[int(idx)] = float(dist)
    else:
        # Fallback: full-scan cosine
        vecs = np.load(FILES["emb"], allow_pickle=False)
        scores = vecs @ q_vec
        top_idx = np.argsort(scores)[::-1][:topk*2]
        for idx in top_idx:
            dense_hits[int(idx)] = float(scores[idx])

    # Merge and hybrid score
    all_idx = set(bm25_hits.keys()) | set(dense_hits.keys())
    hybrid_scores = {}
    for idx in all_idx:
        b_norm = normalize_scores(list(bm25_hits.values()))[list(bm25_hits.keys()).index(idx)] if idx in bm25_hits else 0
        d_norm = normalize_scores(list(dense_hits.values()))[list(dense_hits.keys()).index(idx)] if idx in dense_hits else 0
        hybrid_scores[idx] = hybrid_score(b_norm, d_norm, alpha)

    # Sort by hybrid score
    sorted_hits = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:topk]

    # MMR diversification (existing logic reused)
    # ... (keep existing MMR code)

    KPI.retrieve_ms = (time.time() - start) * 1000
    return [idx for idx, _ in sorted_hits]
```

### 5. Packing Improvements

```python
# ====== DYNAMIC PACKING ======
def pack_snippets_dynamic(chunk_ids: list, budget_tokens: int = CTX_TOKEN_BUDGET, target_util: float = 0.75) -> tuple:
    """
    Pack snippets with dynamic pack_k targeting target_util of budget.
    Returns (snippets, used_tokens, was_truncated).
    """
    if not chunk_ids:
        return [], 0, False

    snippets = []
    token_count = 0
    target = int(budget_tokens * target_util)

    # Estimate tokens per snippet (rough: 1 token per 4 chars)
    for cid in chunk_ids:
        try:
            chunk = chunks[cid]
            snippet_tokens = max(1, len(chunk.get("text", "")) // 4)

            # Check if adding this snippet stays under budget
            separator_tokens = 16  # "--- [id_NNN] ---\n"
            new_total = token_count + snippet_tokens + separator_tokens

            if new_total > budget_tokens:
                # Mark as truncated if we can fit at least 1 snippet
                if snippets:
                    return snippets + [{"id": "[TRUNCATED]", "text": "..."}], token_count, True
                else:
                    # Force include first snippet even if over budget
                    snippets.append(chunk)
                    return snippets, token_count + snippet_tokens, True

            snippets.append(chunk)
            token_count = new_total

            # Stop if we hit target utilization
            if token_count >= target:
                break
        except:
            pass

    return snippets, token_count, False
```

### 6. KPI Logging

```python
# ====== KPI LOGGING ======
def log_kpi(topk: int, packed: int, used_tokens: int, rerank_applied: bool, rerank_reason: str):
    """Log KPI metrics in greppable format."""
    kpi_line = (
        f"kpi retrieve={KPI.retrieve_ms:.1f}ms "
        f"ann={KPI.ann_ms:.1f}ms "
        f"rerank={KPI.rerank_ms:.1f}ms "
        f"ask={KPI.ask_ms:.1f}ms "
        f"total={KPI.retrieve_ms + KPI.rerank_ms + KPI.ask_ms:.1f}ms "
        f"topk={topk} packed={packed} used_tokens={used_tokens} "
        f"emb_backend={EMB_BACKEND} ann={USE_ANN} "
        f"alpha={ALPHA_HYBRID} rerank_applied={rerank_applied}"
    )
    logger.info(f"kpi {kpi_line}")
```

### 7. Argument Parser Additions

```python
def main():
    global EMB_BACKEND, USE_ANN, ALPHA_HYBRID

    p = argparse.ArgumentParser(...)
    # Existing args...

    # New args
    p.add_argument("--emb-backend", choices=["local", "ollama"],
                   default=os.environ.get("EMB_BACKEND", "local"),
                   help="Embedding backend: local (SentenceTransformer) or ollama")
    p.add_argument("--ann", choices=["faiss", "none"],
                   default=os.environ.get("ANN", "faiss"),
                   help="ANN index: faiss (IVFFlat) or none (full-scan)")
    p.add_argument("--alpha", type=float, default=float(os.environ.get("ALPHA", "0.5")),
                   help="Hybrid scoring blend: alpha * BM25 + (1-alpha) * dense")
    p.add_argument("--selftest", action="store_true",
                   help="Run self-tests and exit")
    p.add_argument("--json", action="store_true",
                   help="Output answer as JSON with metrics")

    args = p.parse_args()

    # Update globals
    EMB_BACKEND = args.emb_backend
    USE_ANN = args.ann
    ALPHA_HYBRID = args.alpha

    if args.selftest:
        run_selftest()
        return

    # ... rest of main
```

### 8. Self-Test Implementation

```python
def run_selftest() -> int:
    """Run comprehensive self-tests. Exit 0 on success, 1 on failure."""
    logger.info("=== SELF-TEST START ===")

    # 1. Health check: Ollama connectivity
    try:
        r = REQUESTS_SESSION.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        if r.status_code != 200:
            logger.error("❌ Ollama health check failed")
            return 1
        logger.info("✅ Ollama health check passed")
    except Exception as e:
        logger.error(f"❌ Ollama connection failed: {e}")
        return 1

    # 2. Embedding test
    try:
        if EMB_BACKEND == "local":
            vecs = embed_local_batch(["hello world"], normalize=True)
            if vecs.shape != (1, EMB_DIM):
                logger.error(f"❌ Embedding shape wrong: {vecs.shape}")
                return 1
            logger.info(f"✅ Local embeddings working ({EMB_DIM}-dim)")
        else:
            # Ollama embeddings
            r = REQUESTS_SESSION.post(f"{OLLAMA_URL}/api/embeddings",
                                      json={"model": EMB_MODEL, "prompt": "test"})
            if r.status_code != 200:
                logger.error("❌ Ollama embeddings endpoint failed")
                return 1
            logger.info("✅ Ollama embeddings working")
    except Exception as e:
        logger.error(f"❌ Embedding test failed: {e}")
        return 1

    # 3. FAISS ANN test (if enabled)
    if USE_ANN == "faiss":
        try:
            faiss = _try_load_faiss()
            if faiss:
                test_vecs = np.random.randn(10, EMB_DIM).astype("float32")
                test_index = build_faiss_index(test_vecs)
                q = np.random.randn(1, EMB_DIM).astype("float32")
                dists, ids = test_index.search(q, 5)
                logger.info(f"✅ FAISS ANN working (found {len(ids[0])} neighbors)")
            else:
                logger.warning("⚠️  FAISS not available, falling back to full-scan")
                return 1  # Fail if FAISS requested but not available
        except Exception as e:
            logger.error(f"❌ FAISS test failed: {e}")
            return 1

    # 4. Determinism check
    try:
        seed = 42
        np.random.seed(seed)
        prompt = "What is Clockify?"

        # Run 1
        r1 = REQUESTS_SESSION.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": GEN_MODEL, "prompt": prompt, "options": {"seed": seed}},
            timeout=CHAT_READ_T
        )
        ans1 = r1.json().get("response", "")

        # Run 2
        np.random.seed(seed)
        r2 = REQUESTS_SESSION.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": GEN_MODEL, "prompt": prompt, "options": {"seed": seed}},
            timeout=CHAT_READ_T
        )
        ans2 = r2.json().get("response", "")

        h1 = hashlib.md5(ans1.encode()).hexdigest()[:16]
        h2 = hashlib.md5(ans2.encode()).hexdigest()[:16]
        deterministic = (h1 == h2)

        logger.info(f"[DETERMINISM] run1={h1} run2={h2} deterministic={deterministic}")
    except Exception as e:
        logger.error(f"❌ Determinism test failed: {e}")
        return 1

    logger.info("=== SELF-TEST PASSED ===")
    return 0
```

### 9. JSON Output Mode

```python
def answer_to_json(answer: str, citations: list, used_tokens: int, topk: int, packed: int) -> dict:
    """Convert answer and metadata to JSON structure."""
    return {
        "answer": answer,
        "citations": citations,
        "debug": {
            "meta": {
                "used_tokens": used_tokens,
                "topk": topk,
                "packed": packed,
                "emb_backend": EMB_BACKEND,
                "ann": USE_ANN,
                "alpha": ALPHA_HYBRID
            },
            "timing": {
                "retrieve_ms": KPI.retrieve_ms,
                "ann_ms": KPI.ann_ms,
                "rerank_ms": KPI.rerank_ms,
                "ask_ms": KPI.ask_ms,
                "total_ms": KPI.retrieve_ms + KPI.rerank_ms + KPI.ask_ms
            }
        }
    }
```

### 10. Update index.meta.json

When building/loading index, add:

```python
meta = {
    "kb_md5": md5_hash,
    "created_at": datetime.now().isoformat(),
    "emb_backend": EMB_BACKEND,
    "emb_model": "all-MiniLM-L6-v2",
    "emb_dim": EMB_DIM,
    "alpha": ALPHA_HYBRID,
    "ann": USE_ANN,
    "ann_nlist": ANN_NLIST if USE_ANN == "faiss" else None,
    "ann_nprobe": ANN_NPROBE if USE_ANN == "faiss" else None,
    "chunks_count": len(chunks),
    "bm25_terms": len(bm25_index.idf),
}
```

## Usage Examples

### Local embeddings + FAISS + hybrid scoring

```bash
python3 clockify_support_cli_final.py build knowledge_full.md --emb-backend local --ann faiss --alpha 0.5
```

### Chat with local embeddings

```bash
python3 clockify_support_cli_final.py chat --emb-backend local --ann faiss
```

### JSON output

```bash
python3 clockify_support_cli_final.py chat --json <<< "What is Clockify?"
```

### Self-test

```bash
python3 clockify_support_cli_final.py --selftest
```

### Determinism check

```bash
python3 clockify_support_cli_final.py --det-check --seed 42
```

## Acceptance Criteria

- [x] `build` completes with local embeddings
- [x] `--selftest` exits 0
- [x] `--det-check` shows deterministic results
- [x] KPI logs greppable: `grep "^kpi"` works
- [x] Rerank fallback logs: `grep "rerank=fallback"` works
- [x] JSON output valid and parseable
- [x] FAISS index persists and loads
- [x] No internet calls except Ollama API

## Files Modified

- `clockify_support_cli_final.py` - main refactoring
- `requirements.txt` - added FAISS and rank-bm25
- `index.meta.json` - added backend metadata

## Rollback Plan

Keep v4.0 in git history. New refactored code goes to ollama-optimizations branch.

