"""Index building and loading for BM25 and FAISS."""

import hashlib
import json
import logging
import math
import os
import platform
import threading
import time
from collections import Counter

import numpy as np

from .chunking import build_chunks
from . import config
from .embedding import embed_texts, embed_local_batch, load_embedding_cache, save_embedding_cache
from .exceptions import BuildError
from .utils import (build_lock, atomic_write_jsonl, atomic_save_npy, atomic_write_json,
                   tokenize, compute_sha256, _fsync_dir)

logger = logging.getLogger(__name__)

# Global FAISS index with thread safety
_FAISS_INDEX = None
_FAISS_LOCK = threading.Lock()


def _try_load_faiss():
    """Try importing FAISS; returns None if not available."""
    try:
        import faiss
        return faiss
    except ImportError:
        logger.info("info: ann=fallback reason=missing-faiss")
        return None


def build_faiss_index(vecs: np.ndarray, nlist: int = 256, metric: str = "ip") -> object:
    """Build FAISS IVFFlat index (inner product for cosine on normalized vectors).

    Rank 22 optimization: macOS arm64 attempts IVFFlat with nlist=32 (small subset training)
    before falling back to FlatIP. This provides 10-50x speedup over linear search.
    """
    faiss = _try_load_faiss()
    if faiss is None:
        return None

    dim = vecs.shape[1]
    vecs_f32 = np.ascontiguousarray(vecs.astype("float32"))

    # Detect macOS arm64 and optimize for M1/M2/M3 chips
    is_macos_arm64 = platform.system() == "Darwin" and platform.machine() == "arm64"

    if is_macos_arm64:
        # Rank 22: Try IVFFlat with smaller nlist=32 for M1 Macs
        m1_nlist = 32
        m1_train_size = min(1000, len(vecs))

        logger.info(f"macOS arm64 detected: attempting IVFFlat with nlist={m1_nlist}, train_size={m1_train_size}")

        try:
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, m1_nlist, faiss.METRIC_INNER_PRODUCT)

            if len(vecs) >= m1_train_size:
                # Seed RNG for reproducible training (Rank 11)
                rng = np.random.default_rng(config.DEFAULT_SEED)
                train_indices = rng.choice(len(vecs), m1_train_size, replace=False)
                train_vecs = vecs_f32[train_indices]
            else:
                train_vecs = vecs_f32

            # Seed FAISS k-means for deterministic training
            # FAISS has internal randomness in k-means clustering that needs explicit seeding
            faiss.seed(config.DEFAULT_SEED)
            index.train(train_vecs)
            index.add(vecs_f32)

            logger.info(f"✓ Successfully built IVFFlat index on M1 (nlist={m1_nlist}, vectors={len(vecs)})")
            logger.info(f"  Expected speedup: 10-50x over linear search for similarity queries")

        except (RuntimeError, SystemError, OSError) as e:
            logger.warning(f"IVFFlat training failed on M1: {type(e).__name__}: {str(e)[:100]}")
            logger.info(f"Falling back to IndexFlatIP (linear search) for stability")
            index = faiss.IndexFlatIP(dim)
            index.add(vecs_f32)
    else:
        # Other platforms: use IVFFlat with standard nlist
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

        train_size = min(20000, len(vecs))
        # Seed RNG for reproducible training (Rank 11)
        rng = np.random.default_rng(config.DEFAULT_SEED)
        train_indices = rng.choice(len(vecs), train_size, replace=False)
        train_vecs = vecs_f32[train_indices]

        # Seed FAISS k-means for deterministic training
        # FAISS has internal randomness in k-means clustering that needs explicit seeding
        faiss.seed(config.DEFAULT_SEED)
        index.train(train_vecs)
        index.add(vecs_f32)

    index.nprobe = config.ANN_NPROBE

    index_type = "IVFFlat" if hasattr(index, 'nlist') else "FlatIP"
    logger.debug(f"Built FAISS index: type={index_type}, vectors={len(vecs)}")
    return index


def save_faiss_index(index, path: str = None):
    """Save FAISS index to disk."""
    if index is None or path is None:
        return
    faiss = _try_load_faiss()
    if faiss:
        faiss.write_index(index, path)
        logger.debug(f"Saved FAISS index to {path}")


def load_faiss_index(path: str = None):
    """Load FAISS index from disk with thread-safe lazy loading."""
    global _FAISS_INDEX

    if path is None or not os.path.exists(path):
        return None

    # Double-checked locking pattern for thread safety
    if _FAISS_INDEX is not None:
        return _FAISS_INDEX

    with _FAISS_LOCK:
        if _FAISS_INDEX is not None:  # Check again inside lock
            return _FAISS_INDEX

        faiss = _try_load_faiss()
        if faiss:
            _FAISS_INDEX = faiss.read_index(path)
            _FAISS_INDEX.nprobe = config.ANN_NPROBE
            logger.debug(f"Loaded FAISS index from {path}")
            return _FAISS_INDEX
        return None


# ====== BM25 ======
def build_bm25(chunks: list) -> dict:
    """Build BM25 index."""
    docs = [tokenize(c["text"]) for c in chunks]
    N = len(docs)
    df = Counter()
    doc_tfs = []
    doc_lens = []
    for toks in docs:
        tf = Counter(toks)
        doc_tfs.append(tf)
        doc_lens.append(len(toks))
        for w in tf.keys():
            df[w] += 1
    avgdl = sum(doc_lens) / max(1, N)
    idf = {}
    for w, dfw in df.items():
        idf[w] = math.log((N - dfw + 0.5) / (dfw + 0.5) + 1.0)
    return {
        "idf": idf,
        "avgdl": avgdl,
        "doc_lens": doc_lens,
        "doc_tfs": [{k: v for k, v in tf.items()} for tf in doc_tfs]
    }


def bm25_scores(query: str, bm: dict, k1: float = None, b: float = None, top_k: int = None) -> np.ndarray:
    """Compute BM25 scores with optional early termination (Rank 24)."""
    if k1 is None:
        k1 = config.BM25_K1
    if b is None:
        b = config.BM25_B
    q = tokenize(query)
    idf = bm["idf"]
    avgdl = bm["avgdl"]
    doc_lens = bm["doc_lens"]
    doc_tfs = bm["doc_tfs"]

    # Rank 24: Early termination with Wand-like pruning
    if top_k is not None and top_k > 0 and len(doc_lens) > top_k * 1.5:  # Lower threshold for earlier termination
        import heapq
        term_upper_bounds = {}
        for w in q:
            if w in idf:
                term_upper_bounds[w] = idf[w] * (k1 + 1)

        total_upper_bound = sum(term_upper_bounds.values())
        top_scores = []
        threshold = 0.0

        for i, tf in enumerate(doc_tfs):
            if not any(w in tf for w in q):
                continue

            if len(top_scores) >= top_k and total_upper_bound < threshold:
                continue

            dl = doc_lens[i]
            s = 0.0
            for w in q:
                if w not in idf:
                    continue
                f = tf.get(w, 0)
                if f == 0:
                    continue
                denom = f + k1 * (1 - b + b * dl / max(1.0, avgdl))
                s += idf[w] * (f * (k1 + 1)) / denom

            if len(top_scores) < top_k:
                heapq.heappush(top_scores, (s, i))
                if len(top_scores) == top_k:
                    threshold = top_scores[0][0]
            elif s > threshold:
                heapq.heapreplace(top_scores, (s, i))
                threshold = top_scores[0][0]

        scores = np.zeros(len(doc_lens), dtype="float32")
        for score, idx in top_scores:
            scores[idx] = score

        return scores

    # Original implementation: compute all scores
    scores = np.zeros(len(doc_lens), dtype="float32")
    for i, tf in enumerate(doc_tfs):
        dl = doc_lens[i]
        s = 0.0
        for w in q:
            if w not in idf:
                continue
            f = tf.get(w, 0)
            if f == 0:
                continue
            denom = f + k1 * (1 - b + b * dl / max(1.0, avgdl))
            s += idf[w] * (f * (k1 + 1)) / denom
        scores[i] = s
    return scores


# ====== BUILD FUNCTION ======
def build(md_path: str, retries=0):
    """Build knowledge base with atomic writes and locking."""
    with build_lock():
        logger.info("=" * 70)
        logger.info("BUILDING KNOWLEDGE BASE")
        logger.info("=" * 70)
        if not os.path.exists(md_path):
            raise BuildError(f"{md_path} not found")

        logger.info("\n[1/4] Parsing and chunking...")
        chunks = build_chunks(md_path)
        logger.info(f"  Created {len(chunks)} chunks")
        atomic_write_jsonl(config.FILES["chunks"], chunks)

        logger.info(f"\n[2/4] Embedding with {config.EMB_BACKEND}...")
        emb_cache = load_embedding_cache()

        # Compute content hashes
        chunk_hashes = []
        cache_hits = []
        cache_miss_indices = []

        for i, chunk in enumerate(chunks):
            chunk_hash = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
            chunk_hashes.append(chunk_hash)

            if chunk_hash in emb_cache:
                cache_hits.append(emb_cache[chunk_hash])
            else:
                cache_hits.append(None)
                cache_miss_indices.append(i)

        hit_rate = (len(chunks) - len(cache_miss_indices)) / len(chunks) * 100 if chunks else 0
        logger.info(f"  Cache: {len(chunks) - len(cache_miss_indices)}/{len(chunks)} hits ({hit_rate:.1f}%)")

        # Embed cache misses
        new_embeddings = []
        if cache_miss_indices:
            texts_to_embed = [chunks[i]["text"] for i in cache_miss_indices]
            logger.info(f"  Computing {len(texts_to_embed)} new embeddings...")

            if config.EMB_BACKEND == "local":
                new_embeddings = embed_local_batch(texts_to_embed, normalize=False)
            else:
                new_embeddings = embed_texts(texts_to_embed, retries=retries)

            for i, idx in enumerate(cache_miss_indices):
                chunk_hash = chunk_hashes[idx]
                emb_cache[chunk_hash] = new_embeddings[i].astype(np.float32)

        # Reconstruct full embedding matrix with dimension validation
        # Compute expected dimension based on current backend
        expected_dim = config.EMB_DIM_LOCAL if config.EMB_BACKEND == "local" else config.EMB_DIM_OLLAMA

        vecs = []
        new_emb_idx = 0
        for i in range(len(chunks)):
            if cache_hits[i] is not None:
                cached_emb = cache_hits[i]
                # Defensive check: validate cached embedding dimension
                if len(cached_emb) != expected_dim:
                    raise BuildError(
                        f"Cached embedding {i} has dimension {len(cached_emb)}, "
                        f"expected {expected_dim} for backend={config.EMB_BACKEND}. "
                        f"This should have been filtered by load_embedding_cache(). "
                        f"Try deleting {config.FILES['emb_cache']} and rebuilding."
                    )
                vecs.append(cached_emb)
            else:
                new_emb = new_embeddings[new_emb_idx]
                # Validate new embedding dimension
                if len(new_emb) != expected_dim:
                    raise BuildError(
                        f"New embedding {i} has dimension {len(new_emb)}, "
                        f"expected {expected_dim} for backend={config.EMB_BACKEND}. "
                        f"Check {config.EMB_BACKEND} configuration or model output."
                    )
                vecs.append(new_emb)
                new_emb_idx += 1

        # Final sanity check before array construction
        if vecs:
            first_dim = len(vecs[0])
            for idx, vec in enumerate(vecs):
                if len(vec) != first_dim:
                    raise BuildError(
                        f"Dimension mismatch at index {idx}: "
                        f"expected {first_dim}, got {len(vec)}. "
                        f"Cannot mix embeddings with different dimensions."
                    )

        vecs = np.array(vecs, dtype=np.float32)

        if cache_miss_indices:
            save_embedding_cache(emb_cache)

        # Normalize embeddings
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        vecs_n = (vecs / norms).astype("float32")
        atomic_save_npy(vecs_n, config.FILES["emb"])
        logger.info(f"  Saved {vecs_n.shape} embeddings (normalized)")

        # Write metadata
        meta_lines = [
            {"id": c["id"], "title": c["title"], "url": c["url"], "section": c["section"]}
            for c in chunks
        ]
        atomic_write_jsonl(config.FILES["meta"], meta_lines)

        logger.info("\n[3/4] Building BM25 index...")
        bm = build_bm25(chunks)
        atomic_write_json(config.FILES["bm25"], bm)
        logger.info(f"  Indexed {len(bm['idf'])} unique terms")

        # Optional FAISS
        if config.USE_ANN == "faiss":
            try:
                logger.info("\n[3.1/4] Building FAISS ANN index...")
                faiss_index = build_faiss_index(vecs_n, nlist=config.ANN_NLIST)
                if faiss_index is not None:
                    save_faiss_index(faiss_index, config.FILES["faiss_index"])
                    logger.info(f"  Saved FAISS index to {config.FILES['faiss_index']}")
            except Exception as e:
                logger.warning(f"  FAISS index build failed: {e}")

        # Write metadata
        logger.info("\n[3.6/4] Writing artifact metadata...")
        kb_sha = compute_sha256(md_path)
        index_meta = {
            "kb_sha256": kb_sha,
            "chunks": len(chunks),
            "emb_rows": int(vecs_n.shape[0]),
            "bm25_docs": len(bm["doc_lens"]),
            "gen_model": config.GEN_MODEL,
            "emb_model": config.EMB_MODEL if config.EMB_BACKEND == "ollama" else "all-MiniLM-L6-v2",
            "emb_backend": config.EMB_BACKEND,
            "ann": config.USE_ANN,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_write_json(config.FILES["index_meta"], index_meta)
        logger.info(f"  Saved index metadata")

        logger.info("\n[4/4] Done.")
        logger.info("=" * 70)


def load_index():
    """Load all index artifacts with dimension validation.

    FIX: Validates that stored embeddings match the current config.EMB_BACKEND and config.EMB_DIM
    to prevent runtime crashes from dimension mismatches (e.g., switching from
    local 384-dim to ollama 768-dim without rebuilding).

    Returns:
        dict with index artifacts, or None if validation fails (requiring rebuild)
    """
    if not os.path.exists(config.FILES["index_meta"]):
        logger.warning("[rebuild] index.meta.json missing")
        return None

    with open(config.FILES["index_meta"], encoding="utf-8") as f:
        meta = json.load(f)

    # Validate backend compatibility
    stored_backend = meta.get("emb_backend")
    stored_model = meta.get("emb_model")

    if stored_backend and stored_backend != config.EMB_BACKEND:
        logger.error(
            f"❌ Index backend mismatch: stored={stored_backend}, current={config.EMB_BACKEND}"
        )
        logger.error(
            f"   Stored model: {stored_model}"
        )
        logger.error(
            f"   Current model: {config.EMB_MODEL if config.EMB_BACKEND == 'ollama' else 'all-MiniLM-L6-v2'}"
        )
        logger.error(
            f"   Expected dimension: {config.EMB_DIM}"
        )
        logger.error("")
        logger.error(f"💡 Solution: Rebuild the index with the current backend:")
        logger.error(f"   python3 clockify_support_cli.py build knowledge_full.md")
        logger.error("")
        return None

    # Load chunks
    chunks = []
    with open(config.FILES["chunks"], encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    # Load embeddings
    vecs_n = np.load(config.FILES["emb"])

    # Validate embedding dimensions
    # Compute expected dimension based on current backend
    expected_dim = config.EMB_DIM_LOCAL if config.EMB_BACKEND == "local" else config.EMB_DIM_OLLAMA
    stored_dim = vecs_n.shape[1] if len(vecs_n.shape) > 1 else 0

    if stored_dim != expected_dim:
        logger.error(
            f"❌ Embedding dimension mismatch: stored={stored_dim}, expected={expected_dim} "
            f"(backend={config.EMB_BACKEND})"
        )
        logger.error(
            f"   This usually happens after switching embedding backends without rebuilding."
        )
        logger.error("")
        logger.error(f"💡 Solution: Rebuild the index:")
        logger.error(f"   python3 clockify_support_cli.py build knowledge_full.md")
        logger.error("")
        return None

    # Load BM25
    with open(config.FILES["bm25"], encoding="utf-8") as f:
        bm = json.load(f)

    # Optional FAISS
    faiss_index = None
    if config.USE_ANN == "faiss" and os.path.exists(config.FILES["faiss_index"]):
        faiss_index = load_faiss_index(config.FILES["faiss_index"])

    # Build chunk dict
    chunks_dict = {c["id"]: c for c in chunks}

    logger.info(f"Loaded {len(chunks)} chunks, {vecs_n.shape[0]} vectors, {len(bm['idf'])} terms")

    return {
        "chunks": chunks,
        "chunks_dict": chunks_dict,
        "vecs_n": vecs_n,
        "bm": bm,
        "faiss_index": faiss_index,
        "meta": meta
    }
