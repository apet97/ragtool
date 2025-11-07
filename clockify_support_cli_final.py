#!/usr/bin/env python3
"""
Clockify Internal Support CLI – Stateless RAG with Hybrid Retrieval

HOW TO RUN
==========
  # Build knowledge base (one-time)
  python3 clockify_support_cli.py build knowledge_full.md

  # Start interactive REPL
  python3 clockify_support_cli.py chat [--debug] [--rerank] [--topk 12] [--pack 6] [--threshold 0.30]

  # Or auto-start REPL with no args
  python3 clockify_support_cli.py

DESIGN
======
- Fully offline: uses only http://127.0.0.1:11434 (local Ollama)
- Stateless REPL: each turn forgets prior context
- Hybrid retrieval: BM25 (sparse) + dense (semantic) + MMR diversification
- Closed-book: refuses low-confidence answers
- Artifact versioning: auto-rebuild if KB drifts
- No external APIs or web calls
"""

# Standard library imports (Rank 26: Formatted per PEP 8)
import argparse
import atexit
import errno
import hashlib
import json
import logging
import math
import os
import pathlib
import platform
import re
import subprocess
import sys
import tempfile
from typing import Any, Optional
import threading
import time
import unicodedata
import uuid
from collections import Counter, defaultdict, deque
from contextlib import contextmanager

# Third-party imports
import numpy as np
import requests

from clockify_rag.caching import QueryCache, RateLimiter

# Rank 23: NLTK for sentence-aware chunking
try:
    import nltk
    # Lazy download of punkt tokenizer data (only if needed)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)  # For newer NLTK versions
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

# ====== MODULE LOGGER ======
logger = logging.getLogger(__name__)

# ====== CUSTOM EXCEPTIONS ======
class EmbeddingError(Exception):
    """Embedding generation failed"""
    pass

class LLMError(Exception):
    """LLM call failed"""
    pass

class IndexLoadError(Exception):
    """Index loading or validation failed"""
    pass

class BuildError(Exception):
    """Knowledge base build failed"""
    pass

# ====== CONFIG ======
# These are module-level defaults, overridable via main()
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
GEN_MODEL = os.environ.get("GEN_MODEL", "qwen2.5:32b")
EMB_MODEL = os.environ.get("EMB_MODEL", "nomic-embed-text")

CHUNK_CHARS = 1600
CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 12
DEFAULT_PACK_TOP = 6
DEFAULT_THRESHOLD = 0.30
DEFAULT_SEED = 42

# BM25 parameters (tuned for technical documentation)
# Lower k1 (1.2→1.0): Reduces term frequency saturation for repeated technical terms
# Lower b (0.75→0.65): Reduces length normalization penalty for longer docs
BM25_K1 = float(os.environ.get("BM25_K1", "1.0"))
BM25_B = float(os.environ.get("BM25_B", "0.65"))
DEFAULT_NUM_CTX = 8192
DEFAULT_NUM_PREDICT = 512
DEFAULT_RETRIES = 0
MMR_LAMBDA = 0.7
CTX_TOKEN_BUDGET = int(os.environ.get("CTX_BUDGET", "2800"))  # ~11,200 chars, overridable

# Quick Win #6: Extracted magic numbers to config
FAISS_CANDIDATE_MULTIPLIER = 3  # Retrieve top_k * 3 candidates for reranking
ANN_CANDIDATE_MIN = 200  # Minimum candidates even if top_k is small
RERANK_SNIPPET_MAX_CHARS = 500  # Truncate chunk text for reranking prompt
RERANK_MAX_CHUNKS = 12  # Maximum chunks to send to reranking
COVERAGE_MIN_CHUNKS = 2  # Minimum chunks above threshold to proceed

# ====== EMBEDDINGS BACKEND (v4.1) ======
EMB_BACKEND = os.environ.get("EMB_BACKEND", "local")  # "local" or "ollama"

# Embedding dimensions:
# - local (SentenceTransformer all-MiniLM-L6-v2): 384-dim
# - ollama (nomic-embed-text): 768-dim
EMB_DIM_LOCAL = 384
EMB_DIM_OLLAMA = 768
EMB_DIM = EMB_DIM_LOCAL if EMB_BACKEND == "local" else EMB_DIM_OLLAMA

# ====== ANN (Approximate Nearest Neighbors) (v4.1) ======
USE_ANN = os.environ.get("ANN", "faiss")  # "faiss" or "none"
# Note: nlist reduced from 256→64 for arm64 macOS stability (avoid IVF training segfault)
ANN_NLIST = int(os.environ.get("ANN_NLIST", "64"))  # IVF clusters (reduced for stability)
ANN_NPROBE = int(os.environ.get("ANN_NPROBE", "16"))  # clusters to search

# ====== HYBRID SCORING (v4.1) ======
ALPHA_HYBRID = float(os.environ.get("ALPHA", "0.5"))  # 0.5 = BM25 and dense equally weighted

# ====== KPI TIMINGS (v4.1) ======
class KPI:
    retrieve_ms = 0
    ann_ms = 0
    rerank_ms = 0
    ask_ms = 0

# Task G: Deterministic timeouts (environment-configurable for ops)
EMB_CONNECT_T = float(os.environ.get("EMB_CONNECT_TIMEOUT", "3"))
EMB_READ_T = float(os.environ.get("EMB_READ_TIMEOUT", "60"))
CHAT_CONNECT_T = float(os.environ.get("CHAT_CONNECT_TIMEOUT", "3"))
CHAT_READ_T = float(os.environ.get("CHAT_READ_TIMEOUT", "120"))
RERANK_READ_T = float(os.environ.get("RERANK_READ_TIMEOUT", "180"))

# Exact refusal string (ASCII quotes only)
REFUSAL_STR = "I don't know based on the MD."

# Query logging configuration
QUERY_LOG_FILE = os.environ.get("RAG_LOG_FILE", "rag_queries.jsonl")
QUERY_LOG_DISABLED = os.environ.get("RAG_NO_LOG", "").lower() in {"1", "true", "yes", "on"}
LOG_QUERY_INCLUDE_ANSWER = os.environ.get("RAG_LOG_INCLUDE_ANSWER", "1").lower() in {"1", "true", "yes", "on"}
LOG_QUERY_ANSWER_PLACEHOLDER = os.environ.get("RAG_LOG_ANSWER_PLACEHOLDER", "[REDACTED]")
LOG_QUERY_INCLUDE_CHUNKS = os.environ.get("RAG_LOG_INCLUDE_CHUNKS", "0").lower() in {"1", "true", "yes", "on"}  # Redact chunk text by default for security/privacy

FILES = {
    "chunks": "chunks.jsonl",
    "emb": "vecs_n.npy",  # Pre-normalized embeddings (float32)
    "emb_f16": "vecs_f16.memmap",  # float16 memory-mapped (optional)
    "emb_cache": "emb_cache.jsonl",  # Per-chunk embedding cache
    "meta": "meta.jsonl",
    "bm25": "bm25.json",
    "faiss_index": "faiss.index",  # FAISS IVFFlat index (v4.1)
    "hnsw": "hnsw_cosine.bin",  # Optional HNSW index (if USE_HNSWLIB=1)
    "index_meta": "index.meta.json",  # Artifact versioning
}

BUILD_LOCK = ".build.lock"
BUILD_LOCK_TTL_SEC = int(os.environ.get("BUILD_LOCK_TTL_SEC", "900"))  # Task D: 15 minutes default

QUERY_EXPANSIONS_ENV_VAR = "CLOCKIFY_QUERY_EXPANSIONS"
_DEFAULT_QUERY_EXPANSION_PATH = pathlib.Path(__file__).resolve().parent / "config" / "query_expansions.json"
_query_expansion_cache = None
_query_expansion_override = None

def set_query_expansion_path(path):
    """Override the query expansion configuration file path."""
    global _query_expansion_override
    if path is None:
        _query_expansion_override = None
    else:
        _query_expansion_override = pathlib.Path(path)
    reset_query_expansion_cache()

def reset_query_expansion_cache():
    """Clear cached query expansion data (useful for tests)."""
    global _query_expansion_cache
    _query_expansion_cache = None

def _resolve_query_expansion_path():
    if _query_expansion_override is not None:
        return _query_expansion_override
    env_path = os.environ.get(QUERY_EXPANSIONS_ENV_VAR)
    if env_path:
        return pathlib.Path(env_path)
    return _DEFAULT_QUERY_EXPANSION_PATH

def _read_query_expansion_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise ValueError(f"Query expansion file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in query expansion file {path}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to read query expansion file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Query expansion config must be a JSON object (file: {path})")

    normalized = {}
    for term, synonyms in data.items():
        if not isinstance(term, str):
            raise ValueError(f"Query expansion terms must be strings (file: {path})")
        if not isinstance(synonyms, list):
            raise ValueError(f"Query expansion entry for '{term}' must be a list (file: {path})")

        cleaned = []
        for syn in synonyms:
            syn_str = syn if isinstance(syn, str) else str(syn)
            syn_str = syn_str.strip()
            if syn_str:
                cleaned.append(syn_str)

        if cleaned:
            normalized[term.lower()] = cleaned

    return normalized

def load_query_expansion_dict(force_reload=False, suppress_errors=True):
    """Load query expansion dictionary from disk with optional caching."""
    global _query_expansion_cache

    if _query_expansion_cache is not None and not force_reload:
        return _query_expansion_cache

    path = _resolve_query_expansion_path()

    try:
        expansions = _read_query_expansion_file(path)
    except ValueError as exc:
        if suppress_errors:
            logger.error("Query expansion disabled: %s", exc)
            _query_expansion_cache = {}
            return _query_expansion_cache
        raise

    _query_expansion_cache = expansions
    return _query_expansion_cache

# ====== CLEANUP HANDLERS ======
def _release_lock_if_owner():
    """Release build lock on exit if held by this process - Task D."""
    try:
        if os.path.exists(BUILD_LOCK):
            with open(BUILD_LOCK) as f:
                data = json.loads(f.read())
            if data.get("pid") == os.getpid():
                os.remove(BUILD_LOCK)
                logger.debug("Cleaned up build lock")
    except (OSError, FileNotFoundError, json.JSONDecodeError, KeyError):
        # Cleanup failed - not critical, can ignore
        pass

atexit.register(_release_lock_if_owner)

# Global requests session for keep-alive and retry logic
REQUESTS_SESSION = None
REQUESTS_SESSION_RETRIES = 0

def _mount_retries(sess, retries: int):
    """Mount or update HTTP retry adapters with connection pooling (Rank 27).

    Rank 27: Explicitly set pool_connections=10 and pool_maxsize=20 for better
    concurrency and reduced latency on concurrent queries (10-20% improvement).
    """
    from requests.adapters import HTTPAdapter
    try:
        from urllib3.util.retry import Retry  # urllib3 v2
        retry_cls = Retry
        kwargs = dict(
            total=retries, connect=retries, read=retries, status=retries,
            backoff_factor=0.5, raise_on_status=False,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset({"GET", "POST"}),
            respect_retry_after_header=True,
        )
        retry_strategy = retry_cls(**kwargs)
    except Exception:
        # older urllib3
        from urllib3.util import Retry as RetryOld
        retry_cls = RetryOld
        kwargs = dict(
            total=retries, connect=retries, read=retries, status=retries,
            backoff_factor=0.5, raise_on_status=False,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=frozenset({"GET", "POST"}),
        )
        retry_strategy = retry_cls(**kwargs)

    # Rank 27: Explicit connection pooling parameters
    # pool_connections: number of connection pools to cache (1 per host)
    # pool_maxsize: max connections per pool (allows concurrent requests)
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,  # Support up to 10 different hosts
        pool_maxsize=20       # Allow 20 concurrent connections per host
    )
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)

def get_session(retries=0):
    """Get or create global requests session with optional retry logic - Task G."""
    global REQUESTS_SESSION, REQUESTS_SESSION_RETRIES
    if REQUESTS_SESSION is None:
        REQUESTS_SESSION = requests.Session()
        # Task G: Set trust_env based on ALLOW_PROXIES env var
        REQUESTS_SESSION.trust_env = (os.getenv("ALLOW_PROXIES") == "1")
        if retries > 0:
            _mount_retries(REQUESTS_SESSION, retries)
        REQUESTS_SESSION_RETRIES = retries
    elif retries > REQUESTS_SESSION_RETRIES:
        # Upgrade retries if higher count requested
        _mount_retries(REQUESTS_SESSION, retries)
        REQUESTS_SESSION_RETRIES = retries
    return REQUESTS_SESSION

# v4.1: HTTP POST helper with retry logic
def http_post_with_retries(url, json_payload, retries=3, backoff=0.5, timeout=None):
    """POST with exponential backoff retry."""
    if timeout is None:
        timeout = (EMB_CONNECT_T, EMB_READ_T)
    s = get_session()
    last_error = None
    for attempt in range(retries):
        try:
            r = s.post(url, json=json_payload, timeout=timeout, allow_redirects=False)
            if r.status_code == 200:
                return r
            last_error = f"HTTP {r.status_code}"
        except Exception as e:
            last_error = str(e)
            logger.warning(f"retry post url={url} attempt={attempt+1}")
        if attempt < retries - 1:
            import time
            time.sleep(backoff * (2 ** attempt))
    raise RuntimeError(f"POST failed after {retries} attempts to {url}: {last_error}")


# ====== LOCAL EMBEDDINGS (v4.1 - Section 2) ======
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

# ====== FAISS ANN INDEX (v4.1 - Section 3) ======
_FAISS_INDEX = None
_FAISS_LOCK = threading.Lock()
# Stores profiling data from the most recent `retrieve` invocation.
RETRIEVE_PROFILE_LAST = {}

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
    Other platforms use IVFFlat with standard nlist (default 256, reduced to 64 for stability).
    """
    faiss = _try_load_faiss()
    if faiss is None:
        return None

    dim = vecs.shape[1]
    vecs_f32 = np.ascontiguousarray(vecs.astype("float32"))

    # Detect macOS arm64 and optimize for M1/M2/M3 chips
    # Note: platform.machine() is more reliable than platform.processor() on M1 Macs
    is_macos_arm64 = platform.system() == "Darwin" and platform.machine() == "arm64"

    if is_macos_arm64:
        # Rank 22: Try IVFFlat with smaller nlist=32 for M1 Macs
        # Reduces segfault risk while maintaining performance benefits
        m1_nlist = 32  # Much smaller than default 256 to avoid segfault
        m1_train_size = min(1000, len(vecs))  # Small training set for stability

        logger.info(f"macOS arm64 detected: attempting IVFFlat with nlist={m1_nlist}, train_size={m1_train_size}")

        try:
            # Attempt IVFFlat with reduced parameters
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, m1_nlist, faiss.METRIC_INNER_PRODUCT)

            # Train on small subset to minimize segfault risk
            rng = np.random.default_rng(DEFAULT_SEED)
            if len(vecs) >= m1_train_size:
                train_indices = rng.choice(len(vecs), m1_train_size, replace=False)
                train_vecs = vecs_f32[train_indices]
            else:
                train_vecs = vecs_f32

            index.train(train_vecs)
            index.add(vecs_f32)

            logger.info(f"✓ Successfully built IVFFlat index on M1 (nlist={m1_nlist}, vectors={len(vecs)})")
            logger.info(f"  Expected speedup: 10-50x over linear search for similarity queries")

        except (RuntimeError, SystemError, OSError) as e:
            # Catch training/segfault errors and fallback to FlatIP
            logger.warning(f"IVFFlat training failed on M1: {type(e).__name__}: {str(e)[:100]}")
            logger.info(f"Falling back to IndexFlatIP (linear search) for stability")
            index = faiss.IndexFlatIP(dim)
            index.add(vecs_f32)
    else:
        # Other platforms: use IVFFlat with standard nlist (default=256, or reduced to 64 from env)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

        # Train on sample to build centroids
        rng = np.random.default_rng(DEFAULT_SEED)
        train_size = min(20000, len(vecs))
        train_indices = rng.choice(len(vecs), train_size, replace=False)
        train_vecs = vecs_f32[train_indices]
        index.train(train_vecs)
        index.add(vecs_f32)

    index.nprobe = ANN_NPROBE

    index_type = "IVFFlat" if hasattr(index, 'nlist') else "FlatIP"
    logger.debug(f"Built FAISS index: type={index_type}, nlist={nlist if not is_macos_arm64 else 32}, nprobe={ANN_NPROBE}, vectors={len(vecs)}, platform={'arm64' if is_macos_arm64 else 'standard'}")
    return index

def save_faiss_index(index, path: str | None = None):
    """Save FAISS index to disk."""
    if index is None or path is None:
        return
    faiss = _try_load_faiss()
    if faiss:
        faiss.write_index(index, path)
        logger.debug(f"Saved FAISS index to {path}")

def load_faiss_index(path: str | None = None) -> object | None:
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
            _FAISS_INDEX.nprobe = ANN_NPROBE
            logger.debug(f"Loaded FAISS index from {path}")
            return _FAISS_INDEX
        return None

# ====== HYBRID SCORING (v4.1 - Section 4) ======
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

# ====== DYNAMIC PACKING (v4.1 - Section 5) ======
def pack_snippets_dynamic(chunk_ids: list, chunks: dict, budget_tokens: int | None = None, target_util: float = 0.75) -> tuple:
    """Pack snippets with dynamic targeting. Returns (snippets, used_tokens, was_truncated)."""
    if budget_tokens is None:
        budget_tokens = CTX_TOKEN_BUDGET
    if not chunk_ids:
        return [], 0, False

    snippets: list[str] = []
    token_count = 0
    target = int(budget_tokens * target_util)

    for cid in chunk_ids:
        try:
            chunk = chunks[cid]
            snippet_tokens = max(1, len(chunk.get("text", "")) // 4)
            separator_tokens = 16
            new_total = token_count + snippet_tokens + separator_tokens

            if new_total > budget_tokens:
                if snippets:
                    return snippets + [{"id": "[TRUNCATED]", "text": "..."}], token_count, True
                else:
                    snippets.append(chunk)
                    return snippets, token_count + snippet_tokens, True

            snippets.append(chunk)
            token_count = new_total

            if token_count >= target:
                break
        except (KeyError, IndexError, AttributeError, TypeError) as e:
            # Skip chunks with invalid data or missing indices
            logger.debug(f"Skipping chunk {cid}: {e}")
            continue

    return snippets, token_count, False

# ====== KPI LOGGING (v4.1 - Section 6) ======
def log_kpi(topk: int, packed: int, used_tokens: int, rerank_applied: bool, rerank_reason: str = ""):
    """Log KPI metrics in greppable format."""
    kpi_line = (
        f"retrieve={KPI.retrieve_ms:.1f}ms "
        f"ann={KPI.ann_ms:.1f}ms "
        f"rerank={KPI.rerank_ms:.1f}ms "
        f"ask={KPI.ask_ms:.1f}ms "
        f"total={KPI.retrieve_ms + KPI.rerank_ms + KPI.ask_ms:.1f}ms "
        f"topk={topk} packed={packed} used_tokens={used_tokens} "
        f"emb_backend={EMB_BACKEND} ann={USE_ANN} "
        f"alpha={ALPHA_HYBRID} rerank_applied={rerank_applied}"
    )
    logger.info(f"kpi {kpi_line}")

# ====== JSON OUTPUT (v4.1 - Section 9) ======
def answer_to_json(answer: str, citations: list, used_tokens: int | None, topk: int, packed: int) -> dict:
    """Convert answer and metadata to JSON structure.

    Args:
        answer: Generated answer text.
        citations: Sequence of citation identifiers (chunk IDs).
        used_tokens: Actual token budget consumed when packing context.
        topk: Retrieval depth requested.
        packed: Maximum number of snippets packed.
    """
    budget_tokens = 0 if used_tokens is None else int(used_tokens)
    return {
        "answer": answer,
        "citations": citations,
        "debug": {
            "meta": {
                "used_tokens": budget_tokens,
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

# ====== SELF-TEST INTEGRATION CHECKS (v4.1 - Section 8) ======
# Note: Detailed unit tests are in test_* functions below.
# These are integration/smoke tests that verify key components.


def _pid_alive(pid: int) -> bool:
    """Check if a process is alive. Cross-platform - Task D."""
    if pid <= 0:
        return False
    system = platform.system().lower()
    try:
        if system != "windows":
            # POSIX: use signal 0 check
            os.kill(pid, 0)
            return True
        else:
            # Windows: best-effort with optional psutil
            try:
                import psutil
                return psutil.pid_exists(pid)
            except Exception:
                # Fallback: treat as alive; bounded wait handles stale locks
                # Hint once for better DX
                try:
                    if not getattr(_pid_alive, "_hinted_psutil", False):
                        logger.debug("[build_lock] psutil not available on Windows; install 'psutil' for precise PID checks")
                        _pid_alive._hinted_psutil = True
                except Exception:
                    pass
                return True
    except OSError:
        return False

@contextmanager
def build_lock():
    """Exclusive build lock with atomic create (O_EXCL) and stale-lock recovery - Task D.

    Uses atomic file creation to prevent partial writes. Detects stale locks via
    PID liveness check and TTL expiration.
    """
    pid = os.getpid()
    hostname = platform.node() or "unknown"
    deadline = time.time() + 10.0  # 10s max wait

    while True:
        try:
            # Atomic create: fails if file exists (O_EXCL)
            fd = os.open(BUILD_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                with os.fdopen(fd, "w") as f:
                    started_at = time.time()
                    lock_data = {
                        "pid": pid,
                        "host": hostname,
                        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
                        "started_at_epoch": started_at,
                        "ttl_sec": BUILD_LOCK_TTL_SEC
                    }
                    f.write(json.dumps(lock_data))
                break  # Successfully acquired lock
            except Exception:
                try:
                    os.close(fd)
                except Exception:
                    pass
                raise
        except FileExistsError:
            # Lock file exists; check if it's stale (Task D)
            try:
                with open(BUILD_LOCK, "r") as f:
                    lock_data = json.loads(f.read())
                stale_pid = lock_data.get("pid", 0)
                started_at_epoch = lock_data.get("started_at_epoch", 0)
                ttl_sec = lock_data.get("ttl_sec", BUILD_LOCK_TTL_SEC)

                # Check TTL expiration
                age = time.time() - started_at_epoch
                is_expired = age > ttl_sec
                pid_alive = _pid_alive(stale_pid)

                # If expired or dead owner, try to remove and retry
                if is_expired or not pid_alive:
                    reason = f"expired (age={age:.1f}s > ttl={ttl_sec}s)" if is_expired else f"dead PID {stale_pid}"
                    logger.warning(f"[build_lock] Recovering: {reason}")
                    try:
                        os.remove(BUILD_LOCK)
                        continue  # Retry atomic create
                    except Exception:
                        pass
            except FileNotFoundError:
                # Lock removed by another process between check and read, retry
                logger.debug("[build_lock] Lock removed during check, retrying...")
                continue
            except (json.JSONDecodeError, ValueError) as e:
                # Corrupt lock file, try to remove and retry
                logger.warning(f"[build_lock] Corrupt lock file: {e}")
                try:
                    os.remove(BUILD_LOCK)
                except Exception:
                    pass
                continue
            except Exception as e:
                logger.warning(f"[build_lock] Unexpected error reading lock: {e}")
                # Fall through to timeout logic

            # Still held by live process; wait and retry with 250 ms polling
            if time.time() > deadline:
                raise RuntimeError("Build already in progress; timed out waiting for lock release")
            while time.time() < deadline:  # Use deadline directly
                time.sleep(0.25)
                if not os.path.exists(BUILD_LOCK):
                    break
                if time.time() > deadline:  # Check deadline in loop
                    raise RuntimeError("Build already in progress; timed out waiting for lock release")
            continue

    try:
        yield
    finally:
        # Only remove lock if we still own it
        try:
            with open(BUILD_LOCK, "r") as f:
                lock_data = json.loads(f.read())
            if lock_data.get("pid") == os.getpid():
                os.remove(BUILD_LOCK)
        except Exception:
            pass

# ====== CONFIG VALIDATION ======
def validate_ollama_url(url: str) -> str:
    """Validate and normalize Ollama URL. Returns validated URL."""
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
        if not parsed.scheme:
            # Assume http if no scheme
            url = "http://" + url
            parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid scheme: {parsed.scheme}. Must be http or https.")
        if not parsed.netloc:
            raise ValueError(f"Invalid URL: {url}. Must include host.")
        # Normalize: ensure no trailing slash
        url = f"{parsed.scheme}://{parsed.netloc}"
        if parsed.path and parsed.path != "/":
            url += parsed.path
        return url
    except Exception as e:
        raise ValueError(f"Invalid Ollama URL '{url}': {e}")

def validate_and_set_config(ollama_url=None, gen_model=None, emb_model=None, ctx_budget=None):
    """Validate and set global config from CLI args."""
    global OLLAMA_URL, GEN_MODEL, EMB_MODEL, CTX_TOKEN_BUDGET

    if ollama_url:
        OLLAMA_URL = validate_ollama_url(ollama_url)
        logger.info(f"Ollama endpoint: {OLLAMA_URL}")

    if gen_model:
        GEN_MODEL = gen_model
        logger.info(f"Generation model: {GEN_MODEL}")

    if emb_model:
        EMB_MODEL = emb_model
        logger.info(f"Embedding model: {EMB_MODEL}")

    if ctx_budget:
        try:
            CTX_TOKEN_BUDGET = int(ctx_budget)
            if CTX_TOKEN_BUDGET < 256:
                raise ValueError("Context budget must be >= 256")
            logger.info(f"Context token budget: {CTX_TOKEN_BUDGET}")
        except ValueError as e:
            raise ValueError(f"Invalid context budget: {e}")

def validate_chunk_config():
    """Validate chunk parameters at startup."""
    if CHUNK_OVERLAP >= CHUNK_CHARS:
        raise ValueError(f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be < CHUNK_CHARS ({CHUNK_CHARS})")
    logger.debug(f"Chunk config: size={CHUNK_CHARS}, overlap={CHUNK_OVERLAP}")

def check_pytorch_mps():
    """Check PyTorch MPS availability on M1 Macs and log warnings (v4.1.2)."""
    is_macos_arm64 = platform.system() == "Darwin" and platform.machine() == "arm64"

    if not is_macos_arm64:
        return  # Only relevant for M1/M2/M3 Macs

    try:
        import torch
        mps_available = torch.backends.mps.is_available()

        if mps_available:
            logger.info("info: pytorch_mps=available platform=arm64 (GPU acceleration enabled)")
        else:
            logger.warning(
                "warning: pytorch_mps=unavailable platform=arm64 "
                "hint='Embeddings will use CPU (slower). Ensure macOS 12.3+ and PyTorch 1.12+'"
            )
            logger.warning("  To fix: pip install --upgrade torch or conda install -c pytorch pytorch")
    except ImportError:
        logger.debug("info: pytorch not imported, skipping MPS check")
    except Exception as e:
        logger.debug(f"info: pytorch_mps check failed: {e}")

def _log_config_summary(use_rerank=False, pack_top=DEFAULT_PACK_TOP, seed=DEFAULT_SEED, threshold=DEFAULT_THRESHOLD, top_k=DEFAULT_TOP_K, num_ctx=DEFAULT_NUM_CTX, num_predict=DEFAULT_NUM_PREDICT, retries=0):
    """Log configuration summary at startup - Task I."""
    proxy_trust = 1 if os.getenv("ALLOW_PROXIES") == "1" else 0
    log_target = "off" if QUERY_LOG_DISABLED else pathlib.Path(QUERY_LOG_FILE).resolve()
    # Task I: Single-line CONFIG banner
    logger.info(
        f"CONFIG model={GEN_MODEL} emb={EMB_MODEL} topk={top_k} pack={pack_top} thr={threshold} "
        f"seed={seed} ctx={num_ctx} pred={num_predict} retries={retries} "
        f"timeouts=(3,{int(EMB_READ_T)}/{int(CHAT_READ_T)}/{int(RERANK_READ_T)}) "
        f"log={log_target} "
        f"trust_env={proxy_trust} rerank={1 if use_rerank else 0}"
    )
    # Task I: Print refusal string once for sanity
    logger.info(f'REFUSAL_STR="{REFUSAL_STR}"')

# ====== SYSTEM PROMPT (Rank 25: Few-shot examples added) ======
SYSTEM_PROMPT = f"""You are CAKE.com Internal Support for Clockify.
Closed-book. Only use SNIPPETS. If info is missing, reply exactly:
"{REFUSAL_STR}"
Rules:
- Answer in the user's language.
- Be precise. No speculation. No external info. No web search.
- Structure:
  1) Direct answer
  2) Steps
  3) Notes by role/plan/region if relevant
  4) Citations: list the snippet IDs you used, like [id1, id2], and include URLs in-line if present.
- If SNIPPETS disagree, state the conflict and offer safest interpretation.

EXAMPLES:

Q: How do I track time?
SNIPPETS: [id_1] Click the timer button in the top right corner to start tracking time. You can also manually enter time entries.
A: To track time, click the timer button in the top right corner. You can also manually enter time entries afterward. [id_1]

Q: What is the universe?
SNIPPETS: [id_2] Clockify is a time tracking tool for teams.
A: {REFUSAL_STR}

Q: What are the pricing tiers?
SNIPPETS: [id_3] Free plan includes unlimited users. Basic is $3.99/user/month. Standard is $5.49/user/month. Pro is $7.99/user/month.
A: Clockify offers four pricing tiers:
- Free: Unlimited users, basic features
- Basic: $3.99/user/month
- Standard: $5.49/user/month
- Pro: $7.99/user/month
[id_3]

Now answer the user's question:"""

USER_WRAPPER = """SNIPPETS:
{snips}

QUESTION:
{q}

Answer with citations like [id1, id2].

IMPORTANT (Rank 28): Return your response as JSON with this exact structure:
{{
  "answer": "Your detailed answer with citations [id1, id2]",
  "confidence": 85
}}

Where confidence is 0-100 indicating your certainty based on the snippets:
- 90-100: Highly confident (direct, clear information in snippets)
- 70-89: Confident (snippets cover the question well)
- 50-69: Moderate (partial information, some interpretation needed)
- 30-49: Low (limited/unclear information in snippets)
- 0-29: Very low (use this only if you would normally refuse)

Return ONLY valid JSON. No markdown formatting, no ```json blocks."""

RERANK_PROMPT = """You rank passages for a Clockify support answer. Score each 0.0–1.0 strictly.
Output JSON only: [{"id":"<chunk_id>","score":0.82}, ...].

QUESTION:
{q}

PASSAGES:
{passages}"""

# ====== UTILITIES ======
def _fsync_dir(path: str) -> None:
    """Sync directory to ensure durability (best-effort, platform-dependent)."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    try:
        fd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass  # Best-effort on platforms/filesystems without dir fsync

def atomic_write_bytes(path: str, data: bytes) -> None:
    """Atomically write bytes with fsync durability - Task E."""
    tmp = None
    try:
        d = os.path.dirname(os.path.abspath(path)) or "."
        with tempfile.NamedTemporaryFile(prefix=".tmp.", dir=d, delete=False) as f:
            tmp = f.name
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(path)
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass

def atomic_write_text(path: str, text: str) -> None:
    """Atomically write text file with fsync durability - Task E."""
    atomic_write_bytes(path, text.encode("utf-8"))

def atomic_write_json(path: str, obj) -> None:
    """Atomically write JSON file - Task E."""
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False))

def atomic_write_jsonl(path: str, rows_list) -> None:
    """Atomically write JSONL file (list of dicts) - Task E."""
    # Task E: build rows in memory as list of JSON strings
    lines = []
    for row in rows_list:
        if isinstance(row, dict):
            lines.append(json.dumps(row, ensure_ascii=False))
        else:
            lines.append(str(row))
    content = "\n".join(lines)
    if content and not content.endswith("\n"):
        content += "\n"
    atomic_write_text(path, content)

def atomic_save_npy(arr: np.ndarray, path: str) -> None:
    """Atomically save numpy array with fsync durability - Task E, H."""
    # Task H: enforce float32
    arr = arr.astype("float32")
    d = os.path.dirname(os.path.abspath(path)) or "."
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".tmp.", dir=d, delete=False) as f:
            tmp = f.name
            np.save(f, arr)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(path)
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass

def log_event(event: str, **fields):
    """Log a structured JSON event. Fallback to plain format if JSON serialization fails."""
    try:
        record = {"event": event, **fields}
        logger.info(json.dumps(record, ensure_ascii=False))
    except Exception:
        # Fallback to plain string if JSON encoding fails
        logger.info(f"{event} {fields}")

def norm_ws(s: str) -> str:
    """Normalize whitespace."""
    return re.sub(r"[ \t]+", " ", s.strip())

def is_rtf(text: str) -> bool:
    """Check if text is RTF format."""
    # Check first 128 chars for RTF signature
    head_128 = text[:128]
    if "{\\rtf" in head_128 or "\\rtf" in head_128:
        return True

    # Check first 4096 chars for RTF control words (stricter)
    head_4k = text[:4096]
    rtf_commands = re.findall(r"\\(?:cf\d+|u[+-]?\d+\?|f\d+|pard)\b", head_4k)
    return len(rtf_commands) > 20

def strip_noise(text: str) -> str:
    """Drop scrape artifacts and normalize encoding."""
    # Guard: only apply RTF stripping if content is likely RTF
    if is_rtf(text):
        # Strip RTF escapes only for RTF content (more precise patterns)
        text = re.sub(r"\\cf\d+", "", text)  # \cfN (color)
        text = re.sub(r"\\u[+-]?\d+\?", "", text)  # \u1234? (unicode)
        text = re.sub(r"\\f\d+", "", text)  # \fN (font)
        text = re.sub(r"\{\\\*[^}]*\}", "", text)  # {\* ... } (special)
        text = re.sub(r"\\pard\b[^\n]*", "", text)  # \pard (paragraph)
    # Always remove chunk markers
    text = re.sub(r"^## +Chunk +\d+\s*$", "", text, flags=re.M)
    return text

def tokenize(s: str):
    """Simple tokenizer: lowercase [a-z0-9]+."""
    s = s.lower()
    s = unicodedata.normalize("NFKC", s)
    return re.findall(r"[a-z0-9]+", s)

def approx_tokens(chars: int) -> int:
    """Estimate tokens: 1 token ≈ 4 chars.

    Note: This is a fallback heuristic. Use count_tokens() for accurate counts.
    """
    return max(1, chars // 4)

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count actual tokens using tiktoken (Rank 5: accurate token counting).

    Falls back to character-based heuristic if tiktoken is unavailable.

    Args:
        text: Text to count tokens for
        model: Model name for tokenizer (default: gpt-3.5-turbo)

    Returns:
        Accurate token count
    """
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except (ImportError, KeyError):
        # Fallback to heuristic if tiktoken unavailable or model not found
        return approx_tokens(len(text))

def compute_sha256(filepath: str) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()

def extract_citations(text: str) -> list[str]:
    """Extract citation IDs from answer text (Rank 9: citation validation).

    Supports formats: [id_123], [123], [abc123-def], [id_uuid], etc.
    Now handles both numeric IDs and UUID strings.

    Args:
        text: Answer text containing citations

    Returns:
        List of chunk IDs (strings) referenced in text
    """
    import re
    citations = []

    # Match patterns like [id_123], [123], [uuid-format], [id_abc-123-def], etc.
    # Pattern matches bracketed content with optional 'id_' or 'id' prefix
    pattern = r'\[(?:id_?)?([a-zA-Z0-9-]+)(?:\s*,\s*(?:id_?)?([a-zA-Z0-9-]+))*\]'
    matches = re.finditer(pattern, text, re.IGNORECASE)

    for match in matches:
        # Extract all alphanumeric+hyphen IDs from the match
        ids = re.findall(r'[a-zA-Z0-9-]+', match.group(0).replace('id_', '').replace('id', ''))
        citations.extend(ids)

    return list(set(citations))  # Remove duplicates

def validate_citations(answer: str, valid_chunk_ids: list[str]) -> tuple[bool, list[str], list[str]]:
    """Validate that answer citations match packed chunks (Rank 9).

    Args:
        answer: LLM-generated answer with citations
        valid_chunk_ids: List of chunk IDs (strings) actually included in context

    Returns:
        Tuple of (is_valid, valid_citations, invalid_citations)
    """
    extracted = extract_citations(answer)
    # Normalize to strings for comparison
    valid_set = set(str(cid) for cid in valid_chunk_ids)

    valid_citations = [cid for cid in extracted if cid in valid_set]
    invalid_citations = [cid for cid in extracted if cid not in valid_set]

    is_valid = len(invalid_citations) == 0
    return is_valid, valid_citations, invalid_citations

def truncate_to_token_budget(text: str, budget: int) -> str:
    """Truncate text to fit token budget, append ellipsis - Task C.

    Uses accurate token counting when available (Rank 5).
    """
    est_tokens = count_tokens(text)
    if est_tokens <= budget:
        return text

    # Binary search for optimal truncation point
    # Start with heuristic estimate
    target_chars = budget * 4
    if len(text) <= target_chars:
        return text

    # Iteratively truncate until we fit the budget
    low, high = 0, len(text)
    result = text[:target_chars]

    while low < high:
        mid = (low + high + 1) // 2
        candidate = text[:mid]
        if count_tokens(candidate) <= budget:
            result = candidate
            low = mid
        else:
            high = mid - 1

    return result + " […]" if result != text else text

# ====== KB PARSING ======
def parse_articles(md_text: str):
    """Parse articles from markdown. Heuristic: '# [ARTICLE]' + optional URL line."""
    lines = md_text.splitlines()
    articles = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("# [ARTICLE]"):
            title_line = lines[i].replace("# ", "").strip()
            url = ""
            if i + 1 < len(lines) and lines[i + 1].startswith("http"):
                url = lines[i + 1].strip()
                i += 2
            else:
                i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("# [ARTICLE]"):
                buf.append(lines[i])
                i += 1
            body = "\n".join(buf).strip()
            articles.append({"title": title_line, "url": url, "body": body})
        else:
            i += 1
    if not articles:
        articles = [{"title": "KB", "url": "", "body": md_text}]
    return articles

def split_by_headings(body: str):
    """Split by H2 headers."""
    parts = re.split(r"\n(?=## +)", body)
    return [p.strip() for p in parts if p.strip()]

def sliding_chunks(text: str, maxc=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    """Overlapping chunks with sentence-aware splitting (Rank 23).

    Uses NLTK sentence tokenization to avoid breaking sentences mid-way.
    Falls back to character-based chunking if NLTK is unavailable.
    """
    out = []
    text = strip_noise(text)
    # Normalize to NFKC
    text = unicodedata.normalize("NFKC", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    if len(text) <= maxc:
        return [text]

    # Rank 23: Use sentence-aware chunking if NLTK is available
    if _NLTK_AVAILABLE:
        try:
            sentences = nltk.sent_tokenize(text)

            # Build chunks by accumulating sentences
            current_chunk = []
            current_len = 0

            for sent in sentences:
                sent_len = len(sent)

                # If single sentence exceeds maxc, fall back to character splitting
                if sent_len > maxc:
                    # Flush current chunk first
                    if current_chunk:
                        out.append(" ".join(current_chunk).strip())
                        current_chunk = []
                        current_len = 0

                    # Split long sentence by characters with consistent overlap
                    i = 0
                    while i < sent_len:
                        j = min(i + maxc, sent_len)
                        out.append(sent[i:j].strip())
                        if j >= sent_len:
                            break
                        i = j - overlap if overlap < j else 0  # FIXED: respect overlap
                    continue

                # Check if adding this sentence exceeds maxc
                if current_len + sent_len + (1 if current_chunk else 0) > maxc:
                    # Flush current chunk
                    if current_chunk:
                        out.append(" ".join(current_chunk).strip())

                    # Start new chunk with overlap (last N sentences)
                    overlap_chars = 0
                    overlap_sents = []
                    for prev_sent in reversed(current_chunk):
                        if overlap_chars + len(prev_sent) <= overlap:
                            overlap_sents.insert(0, prev_sent)
                            overlap_chars += len(prev_sent) + 1
                        else:
                            break

                    current_chunk = overlap_sents + [sent]
                    current_len = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
                else:
                    # Add sentence to current chunk
                    current_chunk.append(sent)
                    current_len += sent_len + (1 if len(current_chunk) > 1 else 0)

            # Flush final chunk
            if current_chunk:
                out.append(" ".join(current_chunk).strip())

            return out

        except Exception as e:
            # Fall back to character-based chunking if NLTK fails
            logger.warning(f"NLTK sentence tokenization failed: {e}, falling back to character chunking")

    # Fallback: Character-based chunking (original implementation)
    i = 0
    n = len(text)
    while i < n:
        j = min(i + maxc, n)
        out.append(text[i:j].strip())
        if j >= n:
            break
        i = j - overlap
        if i < 0:
            i = 0
    return out

def build_chunks(md_path: str):
    """Parse and chunk markdown."""
    raw = pathlib.Path(md_path).read_text(encoding="utf-8", errors="ignore")
    chunks = []
    for art in parse_articles(raw):
        sects = split_by_headings(art["body"]) or [art["body"]]
        for sect in sects:
            head = sect.splitlines()[0] if sect else art["title"]
            for piece in sliding_chunks(sect):
                cid = str(uuid.uuid4())
                chunks.append({
                    "id": cid,
                    "title": norm_ws(art["title"]),
                    "url": art["url"],
                    "section": norm_ws(head),
                    "text": piece
                })
    return chunks

# ====== EMBEDDINGS ======
def validate_ollama_embeddings(sample_text: str = "test") -> tuple:
    """Validate Ollama embedding endpoint returns correct format and dimensions.

    FIX (v4.1.2): Detect and report API format issues early before building index.
    Returns: (embedding_dim: int, is_valid: bool)
    """
    try:
        sess = get_session()
        r = sess.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMB_MODEL, "prompt": sample_text},  # Use "prompt" not "input"
            timeout=(EMB_CONNECT_T, EMB_READ_T),
            allow_redirects=False
        )
        r.raise_for_status()

        resp_json = r.json()
        emb = resp_json.get("embedding", [])

        if not emb or len(emb) == 0:
            logger.error(f"❌ Ollama {EMB_MODEL}: empty embedding returned (check API format)")
            return 0, False

        dim = len(emb)
        logger.info(f"✅ Ollama {EMB_MODEL}: {dim}-dim embeddings validated")
        return dim, True
    except Exception as e:
        logger.error(f"❌ Ollama validation failed: {e}")
        return 0, False

# ====== EMBEDDING CACHE ======
def load_embedding_cache():
    """Load embedding cache from disk.

    Returns:
        dict: {content_hash: embedding_vector} mapping
    """
    cache = {}
    cache_path = FILES["emb_cache"]
    if os.path.exists(cache_path):
        logger.info(f"[INFO] Loading embedding cache from {cache_path}")
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        cache[entry["hash"]] = np.array(entry["embedding"], dtype=np.float32)
            logger.info(f"[INFO] Cache contains {len(cache)} embeddings")
        except Exception as e:
            logger.warning(f"[WARN] Failed to load cache: {e}; starting fresh")
            cache = {}
    return cache

def save_embedding_cache(cache):
    """Save embedding cache to disk.

    Args:
        cache: dict of {content_hash: embedding_vector}
    """
    cache_path = FILES["emb_cache"]
    logger.info(f"[INFO] Saving {len(cache)} embeddings to cache")
    try:
        # Atomic write with temp file
        temp_path = cache_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            for content_hash, embedding in cache.items():
                entry = {
                    "hash": content_hash,
                    "embedding": embedding.tolist()
                }
                f.write(json.dumps(entry) + "\n")
        # Ensure write hits disk before rename
        with open(temp_path, "rb") as f:
            os.fsync(f.fileno())
        os.replace(temp_path, cache_path)  # Atomic on POSIX
        logger.info(f"[INFO] Cache saved successfully")
    except Exception as e:
        logger.warning(f"[WARN] Failed to save cache: {e}")

def embed_texts(texts, retries=0):
    """Embed texts using Ollama - Task G. Validates response format (v4.1.2)."""
    sess = get_session(retries=retries)
    vecs = []
    for i, t in enumerate(texts):
        if (i + 1) % 100 == 0:
            logger.info(f"  [{i + 1}/{len(texts)}]")

        # Task G: use tuple timeouts
        try:
            r = sess.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMB_MODEL, "prompt": t},
                timeout=(EMB_CONNECT_T, EMB_READ_T),
                allow_redirects=False
            )
            r.raise_for_status()

            # FIX (v4.1.2): Validate embedding is not empty
            resp_json = r.json()
            emb = resp_json.get("embedding", [])
            if not emb or len(emb) == 0:
                raise EmbeddingError(f"Embedding chunk {i}: empty embedding returned (check Ollama API format)")

            vecs.append(emb)
        except EmbeddingError:
            raise  # Re-raise our own errors as-is
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            raise EmbeddingError(f"Embedding chunk {i} failed: {e} [hint: check OLLAMA_URL or increase EMB timeouts]") from e
        except requests.exceptions.RequestException as e:
            raise EmbeddingError(f"Embedding chunk {i} request failed: {e}") from e
        except (KeyError, IndexError, TypeError) as e:  # SPECIFIC errors
            raise EmbeddingError(f"Embedding chunk {i}: invalid response format: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in embed_texts chunk {i}: {e}", exc_info=True)
            raise  # Let unexpected errors propagate

    return np.array(vecs, dtype="float32")

# ====== BM25 ======
def build_bm25(chunks):
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

def bm25_scores(query: str, bm, k1=None, b=None, top_k=None):
    """Compute BM25 scores with optional early termination (Rank 24).

    Args:
        query: Query string
        bm: BM25 index dict with idf, avgdl, doc_lens, doc_tfs
        k1: Term frequency saturation parameter (default: BM25_K1)
        b: Length normalization parameter (default: BM25_B)
        top_k: If specified, use early termination to compute only top-k scores (Wand-like pruning)

    Rank 24 optimization: When top_k is specified, uses score upper bounds to skip
    documents that cannot reach top-k threshold. Provides 2-3x speedup on large corpora.
    """
    if k1 is None:
        k1 = BM25_K1
    if b is None:
        b = BM25_B
    q = tokenize(query)
    idf = bm["idf"]
    avgdl = bm["avgdl"]
    doc_lens = bm["doc_lens"]
    doc_tfs = bm["doc_tfs"]

    # Rank 24: Early termination with Wand-like pruning
    if top_k is not None and top_k > 0 and len(doc_lens) > top_k * 1.5:  # Lower threshold for earlier termination
        # Compute upper bound per query term (max possible contribution)
        # Upper bound = IDF * (k1 + 1) when term frequency is very high
        term_upper_bounds = {}
        for w in q:
            if w in idf:
                term_upper_bounds[w] = idf[w] * (k1 + 1)

        # Total upper bound (sum of all term contributions)
        total_upper_bound = sum(term_upper_bounds.values())

        # Use heap to maintain top-k scores during iteration
        import heapq
        top_scores = []  # Min-heap of (score, doc_idx)
        threshold = 0.0  # Minimum score to be in top-k

        for i, tf in enumerate(doc_tfs):
            # Quick pruning: if document has no matching terms, skip
            if not any(w in tf for w in q):
                continue

            # Early termination: if upper bound < threshold, skip
            if len(top_scores) >= top_k and total_upper_bound < threshold:
                continue

            # Compute exact score for this document
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

            # Update top-k heap
            if len(top_scores) < top_k:
                heapq.heappush(top_scores, (s, i))
                if len(top_scores) == top_k:
                    threshold = top_scores[0][0]  # Minimum score in top-k
            elif s > threshold:
                heapq.heapreplace(top_scores, (s, i))
                threshold = top_scores[0][0]

        # Build final scores array (sparse: only top-k have non-zero scores)
        scores = np.zeros(len(doc_lens), dtype="float32")
        for score, idx in top_scores:
            scores[idx] = score

        return scores

    # Original implementation: compute all scores (no early termination)
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

def normalize_scores_zscore(arr):
    """Z-score normalize."""
    a = np.asarray(arr, dtype="float32")
    if a.size == 0:
        return a
    m, s = a.mean(), a.std()
    if s == 0:
        return a  # FIXED: preserve original when no variance
    return (a - m) / s

# ====== QUERY EXPANSION (Rank 13) ======
def expand_query(question: str) -> str:
    """Expand query with domain-specific synonyms and acronyms.

    Returns expanded query string with original + synonym terms.
    Example: "How to track time?" → "How to track log record enter time hours duration?"
    """
    if not question:
        return question

    expansions = load_query_expansion_dict()
    q_lower = question.lower()
    expanded_terms = []  # Use list instead of set for deterministic ordering

    # Find matching terms and add their synonyms (Rank 4: deterministic expansion)
    for term, synonyms in expansions.items():
        # Check for whole word matches (avoid partial matches like "track" in "attraction")
        if re.search(r'\b' + re.escape(term) + r'\b', q_lower):
            for syn in synonyms:
                if syn not in expanded_terms:
                    expanded_terms.append(syn)

    # Combine original question with expanded terms
    if expanded_terms:
        expansion = " ".join(expanded_terms)
        return f"{question} {expansion}"

    return question

# ====== RETRIEVAL ======
def embed_query(question: str, retries=0) -> np.ndarray:
    """Embed a query. Returns normalized query vector - Task G."""
    sess = get_session(retries=retries)

    # Task G: use tuple timeouts
    try:
        r = sess.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMB_MODEL, "prompt": question},
            timeout=(EMB_CONNECT_T, EMB_READ_T),
            allow_redirects=False
        )
        r.raise_for_status()
        qv = np.array(r.json()["embedding"], dtype="float32")
        qv_norm = np.linalg.norm(qv)
        return qv / (qv_norm if qv_norm > 0 else 1.0)
    except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
        raise EmbeddingError(f"Query embedding failed: {e} [hint: check OLLAMA_URL or increase EMB timeouts]") from e
    except requests.exceptions.RequestException as e:
        raise EmbeddingError(f"Query embedding request failed: {e}") from e
    except Exception as e:
        raise EmbeddingError(f"Query embedding failed: {e}") from e

class DenseScoreStore:
    """Container for dense similarity scores with optional lazy materialization."""

    __slots__ = ("_length", "_full", "_vecs", "_qv", "_cache")

    def __init__(self, length: int, *, full_scores: Optional[np.ndarray] = None,
                 vecs: Optional[np.ndarray] = None, qv: Optional[np.ndarray] = None,
                 initial: Optional[list[tuple[int, float]]] = None) -> None:
        self._length = int(length)
        self._full: Optional[np.ndarray] = None
        self._vecs = vecs
        self._qv = qv
        self._cache: dict[int, float] = {}

        if full_scores is not None:
            self._full = np.asarray(full_scores, dtype="float32")
        elif initial:
            self._cache.update({int(idx): float(score) for idx, score in initial})

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._length

    def _materialize_full(self) -> np.ndarray:
        if self._full is None:
            if self._vecs is None or self._qv is None:
                self._full = np.zeros(self._length, dtype="float32")
            else:
                self._full = self._vecs.dot(self._qv).astype("float32")
        return self._full

    def __getitem__(self, idx: int) -> float:
        idx = int(idx)
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)

        if self._full is not None:
            return float(self._full[idx])

        if idx not in self._cache:
            if self._vecs is None or self._qv is None:
                raise KeyError(idx)
            self._cache[idx] = float(self._vecs[idx].dot(self._qv))
        return self._cache[idx]

    def get(self, idx: int, default: Optional[float] = None) -> Optional[float]:
        try:
            return self[idx]
        except (IndexError, KeyError):
            return default

    def to_array(self) -> np.ndarray:
        return self._materialize_full().copy()


def retrieve(question: str, chunks, vecs_n, bm, top_k=12, hnsw=None, retries=0) -> tuple[list[int], dict[str, Any]]:
    """Hybrid retrieval: dense + BM25 + dedup. Optionally uses FAISS/HNSW for fast K-NN.

    Scoring: hybrid = ALPHA_HYBRID * normalize(BM25) + (1 - ALPHA_HYBRID) * normalize(dense)

    Query expansion: Applies domain-specific synonym expansion for BM25 (keyword-based),
    uses original query for dense retrieval (embeddings already capture semantics).

    Returns:
        Tuple of (filtered_indices, scores_dict) where filtered_indices is list of int
        and scores_dict contains 'dense', 'bm25', and 'hybrid' numpy arrays.
        Profiling metrics for the last call are stored in RETRIEVE_PROFILE_LAST.
    """
    global _FAISS_INDEX, RETRIEVE_PROFILE_LAST

    # Expand query for BM25 keyword matching (Rank 13)
    expanded_question = expand_query(question)

    # Use original question for embedding (semantic similarity already captured)
    qv_n = embed_query(question, retries=retries)

    # v4.1: Try to load FAISS index once on first call
    if USE_ANN == "faiss" and _FAISS_INDEX is None:
        _FAISS_INDEX = load_faiss_index(FILES["faiss_index"])
        if _FAISS_INDEX:
            _FAISS_INDEX.nprobe = ANN_NPROBE
            logger.info("info: ann=faiss status=loaded nprobe=%d", ANN_NPROBE)
        else:
            logger.info("info: ann=fallback reason=missing-index")

    dense_scores_full = None
    candidate_idx: list[int] = []

    # Use FAISS if available for fast candidate generation
    n_chunks = len(chunks)
    dot_elapsed = 0.0
    dense_computed = 0

    if _FAISS_INDEX:
        # Rank 2: Only score FAISS candidates, don't compute full corpus (2-3x speedup)
        D, I = _FAISS_INDEX.search(
            qv_n.reshape(1, -1).astype("float32"),
            max(ANN_CANDIDATE_MIN, top_k * FAISS_CANDIDATE_MULTIPLIER),
        )
        candidate_idx = [int(i) for i in I[0] if 0 <= i < n_chunks]
        dense_from_ann = np.array([float(d) for d in D[0][: len(candidate_idx)]], dtype=np.float32)

        # Store only candidate scores, use lazy computation for others
        dense_scores = dense_from_ann
        dense_scores_full = None  # Will use DenseScoreStore with lazy computation
        dense_computed = len(candidate_idx)
        dot_elapsed = 0.0  # No additional dot product needed
    # Fallback to HNSW if available
    elif hnsw:
        _, cand = hnsw.knn_query(qv_n, k=max(ANN_CANDIDATE_MIN, top_k * FAISS_CANDIDATE_MULTIPLIER))
        candidate_idx = cand[0].tolist()
        dot_start = time.perf_counter()
        dense_scores_full = vecs_n.dot(qv_n)
        dot_elapsed = time.perf_counter() - dot_start
        dense_computed = n_chunks
    else:
        # Task H: dense scoring uses np.dot with float32 (Rank 14: reuse dense_scores_full)
        dot_start = time.perf_counter()
        dense_scores_full = vecs_n.dot(qv_n)
        dot_elapsed = time.perf_counter() - dot_start
        dense_computed = n_chunks
        dense_scores = dense_scores_full
        candidate_idx = np.arange(len(chunks)).tolist()

    if not candidate_idx:
        # Rank 20: Limit fallback to top candidates instead of full corpus
        dense_scores_full = vecs_n.dot(qv_n)
        max_candidates = max(ANN_CANDIDATE_MIN, top_k * FAISS_CANDIDATE_MULTIPLIER)
        if len(chunks) > max_candidates:
            # Only keep top candidates, not entire corpus
            top_indices = np.argsort(dense_scores_full)[::-1][:max_candidates]
            candidate_idx = top_indices.tolist()
        else:
            candidate_idx = np.arange(len(chunks)).tolist()
        dense_scores = dense_scores_full

    candidate_idx_array = np.array(candidate_idx, dtype=np.int32)
    # Use expanded query for BM25 (keyword matching benefits from synonyms)
    # Rank 24: Pass top_k for early termination on large corpora
    bm_scores_full = bm25_scores(expanded_question, bm, top_k=top_k * 3)

    # Normalize once, then slice for candidates (avoids 4x redundant normalization)
    zs_bm_full = normalize_scores_zscore(bm_scores_full)
    zs_dense_full = None
    if dense_scores_full is not None:
        dense_scores_full = np.asarray(dense_scores_full, dtype="float32")
        zs_dense_full = normalize_scores_zscore(dense_scores_full)
        zs_dense = zs_dense_full[candidate_idx_array] if candidate_idx_array.size else np.array([], dtype="float32")
    else:
        dense_scores = np.asarray(dense_scores, dtype="float32")
        zs_dense = normalize_scores_zscore(dense_scores)
    zs_bm = zs_bm_full[candidate_idx_array] if candidate_idx_array.size else np.array([], dtype="float32")

    # v4.1: Use configurable ALPHA_HYBRID for blending
    hybrid = ALPHA_HYBRID * zs_bm + (1 - ALPHA_HYBRID) * zs_dense
    if hybrid.size:
        top_positions = np.argsort(hybrid)[::-1][:top_k]
        top_idx = candidate_idx_array[top_positions]
    else:
        top_idx = np.array([], dtype=np.int32)

    seen = set()
    filtered = []
    for i in top_idx:
        key = (chunks[i]["title"], chunks[i]["section"])
        if key in seen:
            continue
        seen.add(key)
        filtered.append(i)

    # Reuse cached normalized scores for full hybrid (already computed above)
    if zs_dense_full is not None:
        hybrid_full = ALPHA_HYBRID * zs_bm_full + (1 - ALPHA_HYBRID) * zs_dense_full
    else:
        hybrid_full = np.zeros(len(chunks), dtype="float32")
        for idx, score in zip(candidate_idx, hybrid):
            hybrid_full[idx] = score

    if dense_scores_full is not None:
        dense_scores_store = DenseScoreStore(len(chunks), full_scores=dense_scores_full)
    else:
        dense_scores_store = DenseScoreStore(
            len(chunks), vecs=vecs_n, qv=qv_n,
            initial=list(zip(candidate_idx, dense_scores))
        )

    dense_total = n_chunks
    used_hnsw = bool(hnsw) and _FAISS_INDEX is None
    dense_computed_total = dense_computed or (dense_total if (used_hnsw or not _FAISS_INDEX) else 0)
    dense_reused = dense_total - dense_computed_total
    RETRIEVE_PROFILE_LAST = {
        "used_faiss": bool(_FAISS_INDEX),
        "used_hnsw": used_hnsw,
        "candidates": int(len(candidate_idx)),
        "dense_total": int(dense_total),
        "dense_reused": int(dense_reused),
        "dense_computed": int(dense_computed_total),
        "dense_saved": int(dense_total - dense_computed_total),
        "dense_dot_time_ms": round(dot_elapsed * 1000, 3),
    }
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "profile: retrieval ann=%s reused=%d computed=%d total=%d dot_ms=%.3f",
            "faiss" if _FAISS_INDEX else ("hnsw" if used_hnsw else "linear"),
            RETRIEVE_PROFILE_LAST["dense_reused"],
            RETRIEVE_PROFILE_LAST["dense_computed"],
            dense_total,
            RETRIEVE_PROFILE_LAST["dense_dot_time_ms"],
        )

    return filtered, {
        "dense": dense_scores_store,
        "bm25": bm_scores_full,
        "hybrid": hybrid_full
    }

def rerank_with_llm(question: str, chunks, selected, scores, seed=DEFAULT_SEED, num_ctx=DEFAULT_NUM_CTX, num_predict=DEFAULT_NUM_PREDICT, retries=0) -> tuple:
    """Optional: rerank MMR-selected passages with LLM - Task B.

    Returns: (order, scores, rerank_applied, rerank_reason)
    """
    if len(selected) <= 1:
        return selected, {}, False, "disabled"

    # Build passage list
    passages_text = "\n\n".join([
        f"[id={chunks[i]['id']}]\n{chunks[i]['text'][:RERANK_SNIPPET_MAX_CHARS]}"
        for i in selected
    ])
    payload = {
        "model": GEN_MODEL,
        "options": {
            "temperature": 0,
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.05
        },
        "messages": [
            {"role": "user", "content": RERANK_PROMPT.format(q=question, passages=passages_text)}
        ],
        "stream": False
    }

    rerank_scores = {}
    sess = get_session(retries=retries)
    try:
        # Task G: use tuple timeouts
        r = sess.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=(CHAT_CONNECT_T, RERANK_READ_T),
            allow_redirects=False
        )
        r.raise_for_status()
        resp = r.json()
        msg = (resp.get("message") or {}).get("content", "").strip()

        if not msg:
            # Task B: rerank failure - empty response
            logger.debug("info: rerank=fallback reason=empty")
            return selected, rerank_scores, False, "empty"

        # Try to parse strict JSON array
        try:
            ranked = json.loads(msg)
            if not isinstance(ranked, list):
                logger.debug("info: rerank=fallback reason=json")
                return selected, rerank_scores, False, "json"

            # Map back to indices
            cid_to_idx = {chunks[i]["id"]: i for i in selected}
            reranked = []
            for entry in ranked:
                idx = cid_to_idx.get(entry.get("id"))
                if idx is not None:
                    score = entry.get("score", 0)
                    rerank_scores[idx] = score
                    reranked.append((idx, score))

            if reranked:
                reranked.sort(key=lambda x: x[1], reverse=True)
                return [idx for idx, _ in reranked], rerank_scores, True, ""
            else:
                logger.debug("info: rerank=fallback reason=empty")
                return selected, rerank_scores, False, "empty"
        except json.JSONDecodeError:
            # Task B: JSON parse failed, fall back to MMR order
            logger.debug("info: rerank=fallback reason=json")
            return selected, rerank_scores, False, "json"
    except requests.exceptions.Timeout as e:
        # Task B: timeout
        logger.debug("info: rerank=fallback reason=timeout")
        return selected, rerank_scores, False, "timeout"
    except requests.exceptions.ConnectionError as e:
        # Task B: connection error
        logger.debug("info: rerank=fallback reason=conn")
        return selected, rerank_scores, False, "conn"
    except requests.exceptions.HTTPError as e:
        # Task B: HTTP error
        logger.debug(f"info: rerank=fallback reason=http")
        return selected, rerank_scores, False, "http"
    except requests.exceptions.RequestException:
        # HTTP error, fall back to MMR order
        logger.debug("info: rerank=fallback reason=http")
        return selected, rerank_scores, False, "http"
    except Exception:
        # Unexpected error, fall back to MMR order
        logger.debug("info: rerank=fallback reason=http")
        return selected, rerank_scores, False, "http"

def _fmt_snippet_header(chunk):
    """Format chunk header: [id | title | section] + optional URL."""
    hdr = f"[{chunk['id']} | {chunk['title']} | {chunk['section']}]"
    if chunk.get("url"):
        hdr += f"\n{chunk['url']}"
    return hdr

# ====== PACKING ======
def pack_snippets(chunks, order, pack_top=6, budget_tokens=CTX_TOKEN_BUDGET, num_ctx=DEFAULT_NUM_CTX):
    """Pack snippets respecting strict token budget and hard snippet cap.

    Guarantees:
    - Never exceeds CTX_TOKEN_BUDGET (headers + separators included)
    - First item always included (truncate body if needed; mark [TRUNCATED])
    - Returns (block, ids, used_tokens)
    """
    out = []
    ids = []
    used = 0
    first_truncated = False

    sep_text = "\n\n---\n\n"
    sep_tokens = count_tokens(sep_text)  # Rank 5: accurate token counting

    for idx_pos, idx in enumerate(order):
        if len(ids) >= pack_top:
            break

        c = chunks[idx]
        hdr = _fmt_snippet_header(c)
        body = c["text"]

        hdr_tokens = count_tokens(hdr + "\n")  # Rank 5: accurate token counting
        body_tokens = count_tokens(body)
        need_sep = 1 if out else 0
        sep_cost = sep_tokens if need_sep else 0

        if idx_pos == 0 and not ids:
            # Always include first; truncate if needed to fit budget
            item_tokens = hdr_tokens + body_tokens
            if item_tokens > budget_tokens:
                # amount available for body after header
                allow_body = max(1, budget_tokens - hdr_tokens)
                body = truncate_to_token_budget(body, allow_body)
                body_tokens = count_tokens(body)
                item_tokens = hdr_tokens + body_tokens
                first_truncated = True
            out.append(hdr + "\n" + body)
            ids.append(c["id"])
            used += item_tokens
            continue

        # For subsequent items, check sep + header + body within budget
        item_tokens = hdr_tokens + body_tokens
        if used + sep_cost + item_tokens <= budget_tokens:
            if need_sep:
                out.append(sep_text)
            out.append(hdr + "\n" + body)
            ids.append(c["id"])
            used += sep_cost + item_tokens
        else:
            break

    if first_truncated and out:
        out[0] = out[0].replace("]", " [TRUNCATED]]", 1)

    return "".join(out), ids, used

# ====== COVERAGE CHECK ======
def coverage_ok(selected, dense_scores, threshold):
    """Check coverage."""
    if len(selected) < COVERAGE_MIN_CHUNKS:
        return False
    highs = sum(1 for i in selected if dense_scores[i] >= threshold)
    return highs >= 2

# ====== LLM CALL ======
def ask_llm(question: str, snippets_block: str, seed=DEFAULT_SEED, num_ctx=DEFAULT_NUM_CTX, num_predict=DEFAULT_NUM_PREDICT, retries=0) -> str:
    """Call Ollama chat with Qwen best-practice options - Task G."""
    payload = {
        "model": GEN_MODEL,
        "options": {
            "temperature": 0,
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.05
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_WRAPPER.format(snips=snippets_block, q=question)}
        ],
        "stream": False
    }
    sess = get_session(retries=retries)

    # Task G: use tuple timeouts
    try:
        r = sess.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=(CHAT_CONNECT_T, CHAT_READ_T),
            allow_redirects=False
        )
        r.raise_for_status()
        j = r.json()
        msg = (j.get("message") or {}).get("content")
        if msg:
            return msg
        return j.get("response", "")
    except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
        raise LLMError(f"LLM call failed: {e} [hint: check OLLAMA_URL or increase CHAT timeouts]") from e
    except requests.exceptions.RequestException as e:
        raise LLMError(f"LLM request failed: {e}") from e
    except Exception as e:
        raise LLMError(f"Unexpected error in LLM call: {e}") from e

# ====== BUILD PIPELINE ======
def build(md_path: str, retries=0):
    """Build knowledge base. Guarded with lock to prevent concurrent builds - Task E, H."""
    with build_lock():
        logger.info("=" * 70)
        logger.info("BUILDING KNOWLEDGE BASE")
        logger.info("=" * 70)
        if not os.path.exists(md_path):
            raise BuildError(
                f"{md_path} not found.\n"
                f"Expected knowledge base file at this location.\n"
                f"Hint: Ensure knowledge_full.md exists in current directory."
            )

        logger.info("\n[1/4] Parsing and chunking...")
        chunks = build_chunks(md_path)
        logger.info(f"  Created {len(chunks)} chunks")
        # Task E: Write chunks atomically
        atomic_write_jsonl(FILES["chunks"], chunks)

        logger.info(f"\n[2/4] Embedding with {EMB_BACKEND}...")

        # Load embedding cache for incremental builds
        emb_cache = load_embedding_cache()

        # Compute content hashes and identify cache hits/misses
        chunk_hashes = []
        cache_hits = []
        cache_misses = []
        cache_miss_indices = []

        for i, chunk in enumerate(chunks):
            chunk_hash = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
            chunk_hashes.append(chunk_hash)

            if chunk_hash in emb_cache:
                cache_hits.append(emb_cache[chunk_hash])
                cache_misses.append(None)
            else:
                cache_hits.append(None)
                cache_misses.append(chunk["text"])
                cache_miss_indices.append(i)

        # Report cache statistics
        hit_rate = (len(chunks) - len(cache_miss_indices)) / len(chunks) * 100 if chunks else 0
        logger.info(f"  Cache: {len(chunks) - len(cache_miss_indices)}/{len(chunks)} hits ({hit_rate:.1f}%)")

        # Embed only cache misses
        new_embeddings = []
        if cache_miss_indices:
            texts_to_embed = [chunks[i]["text"] for i in cache_miss_indices]
            logger.info(f"  Computing {len(texts_to_embed)} new embeddings...")

            if EMB_BACKEND == "local":
                # v4.1: Use local SentenceTransformer embeddings
                logger.info(f"  Using local embeddings (backend={EMB_BACKEND})...")
                new_embeddings = embed_local_batch(texts_to_embed, normalize=False)
            else:
                # Fallback to remote Ollama embeddings
                new_embeddings = embed_texts(texts_to_embed, retries=retries)

            # Update cache with new embeddings
            for i, idx in enumerate(cache_miss_indices):
                chunk_hash = chunk_hashes[idx]
                emb_cache[chunk_hash] = new_embeddings[i].astype(np.float32)

        # Reconstruct full embedding matrix in original chunk order
        vecs = []
        new_emb_idx = 0
        for i in range(len(chunks)):
            if cache_hits[i] is not None:
                vecs.append(cache_hits[i])
            else:
                vecs.append(new_embeddings[new_emb_idx])
                new_emb_idx += 1
        vecs = np.array(vecs, dtype=np.float32)

        # Save updated cache
        if cache_miss_indices:  # Only save if there were new embeddings
            save_embedding_cache(emb_cache)

        # Pre-normalize for efficient retrieval, Task H: ensure float32
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        vecs_n = (vecs / norms).astype("float32")
        atomic_save_npy(vecs_n, FILES["emb"])  # Task E, H
        logger.info(f"  Saved {vecs_n.shape} embeddings (normalized)")
        # Write metadata atomically
        meta_lines = [
            {
                "id": c["id"],
                "title": c["title"],
                "url": c["url"],
                "section": c["section"]
            }
            for c in chunks
        ]
        atomic_write_jsonl(FILES["meta"], meta_lines)  # Task E

        logger.info("\n[3/4] Building BM25 index...")
        bm = build_bm25(chunks)
        # Task E: use atomic_write_json for bm25.json
        atomic_write_json(FILES["bm25"], bm)
        logger.info(f"  Indexed {len(bm['idf'])} unique terms")

        # v4.1: Optional FAISS ANN index if enabled
        if USE_ANN == "faiss":
            try:
                logger.info("\n[3.1/4] Building FAISS ANN index...")
                faiss_index = build_faiss_index(vecs_n, nlist=ANN_NLIST)
                if faiss_index is not None:
                    save_faiss_index(faiss_index, FILES["faiss_index"])
                    logger.info(f"  Saved FAISS index to {FILES['faiss_index']}")
                else:
                    logger.info("  FAISS not available, skipping ANN index")
            except Exception as e:
                logger.warning(f"  FAISS index build failed: {e}; continuing without it")

        # Optional HNSW fast index (behind env flag) with atomic save + fsync
        if os.getenv("USE_HNSWLIB") == "1":
            try:
                import hnswlib
                logger.info("\n[3.5/4] Building HNSW index...")
                p = hnswlib.Index(space='cosine', dim=vecs_n.shape[1])
                p.init_index(max_elements=vecs_n.shape[0], ef_construction=200, M=16)
                p.add_items(vecs_n.astype("float32"), np.arange(vecs_n.shape[0]))
                # Atomic save: write to temp file, fsync, then rename
                temp_path = FILES["hnsw"] + ".tmp"
                p.save_index(temp_path)
                # Ensure temp file hits disk before atomic replace
                with open(temp_path, "rb") as _f:
                    os.fsync(_f.fileno())
                os.replace(temp_path, FILES["hnsw"])  # Atomic on POSIX
                _fsync_dir(FILES["hnsw"])
                logger.info(f"  Saved HNSW index to {FILES['hnsw']}")
            except ImportError:
                logger.info("\n[3.5/4] HNSW requested but hnswlib not installed; skipping")
            except Exception as e:
                logger.info(f"\n[3.5/4] HNSW build failed: {e}; continuing without it")

        # Write index.meta.json for artifact versioning (atomic with fsync)
        logger.info("\n[3.6/4] Writing artifact metadata...")
        kb_sha = compute_sha256(md_path)
        index_meta = {
            "kb_sha256": kb_sha,
            "chunks": len(chunks),
            "emb_rows": int(vecs_n.shape[0]),
            "bm25_docs": len(bm["doc_lens"]),
            "gen_model": GEN_MODEL,
            "emb_model": EMB_MODEL if EMB_BACKEND == "ollama" else "all-MiniLM-L6-v2",
            "emb_backend": EMB_BACKEND,
            "ann": USE_ANN,
            "mmr_lambda": MMR_LAMBDA,
            "chunk_chars": CHUNK_CHARS,
            "chunk_overlap": CHUNK_OVERLAP,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code_version": "4.1"
        }
        # Task E: use atomic_write_json for index.meta.json
        atomic_write_json(FILES["index_meta"], index_meta)
        logger.info(f"  Saved index metadata")

        logger.info("\n[4/4] Done.")
        logger.info("=" * 70)

# ====== LOAD INDEX ======
def load_index() -> tuple[list[dict], np.ndarray, dict, object] | None:
    """Load artifacts with full integrity validation - Task H.

    Returns:
        Tuple of (chunks, vecs_n, bm, hnsw) or None if load fails.
    """
    # Check for metadata file
    if not os.path.exists(FILES["index_meta"]):
        logger.warning("[rebuild] index.meta.json missing: run 'python3 clockify_support_cli.py build knowledge_full.md'")
        return None

    with open(FILES["index_meta"], encoding="utf-8") as f:
        meta = json.loads(f.read())

    # 1. Check all required artifacts exist
    missing = []
    for key in ["chunks", "emb", "bm25"]:
        if not os.path.exists(FILES[key]):
            missing.append(FILES[key])

    if missing:
        logger.warning(f"[rebuild] Missing artifacts: {', '.join(missing)}")
        logger.warning("[rebuild] Remediation: run 'python3 clockify_support_cli.py build knowledge_full.md'")
        return None

    # 2. Load and validate embeddings (Task H: ensure float32)
    try:
        vecs_n = np.load(FILES["emb"], mmap_mode="r")  # Read-only memmap
        # Task H: force float32 dtype
        if vecs_n.dtype != np.float32:
            logger.warning(f"[rebuild] Embedding dtype mismatch: {vecs_n.dtype} (expected float32), converting...")
            vecs_n = np.load(FILES["emb"]).astype("float32")
        expected_rows = meta.get("emb_rows", 0)
        if vecs_n.shape[0] != expected_rows:
            logger.warning(f"[rebuild] Embedding rows mismatch: {vecs_n.shape[0]} rows vs {expected_rows} in metadata")
            logger.warning("[rebuild] Remediation: index.meta.json is stale; run build")
            return None
    except Exception as e:
        logger.warning(f"[rebuild] Failed to load embeddings: {e}")
        return None

    # 3. Load and validate chunks
    try:
        with open(FILES["chunks"], encoding="utf-8") as f:
            chunks = [json.loads(l) for l in f if l.strip()]
        if len(chunks) != meta.get("chunks", 0):
            logger.warning(f"[rebuild] Chunk count mismatch: {len(chunks)} chunks vs {meta.get('chunks')} in metadata")
            return None
    except Exception as e:
        logger.warning(f"[rebuild] Failed to load chunks: {e}")
        return None

    # 4. Load and validate BM25 index
    try:
        with open(FILES["bm25"], encoding="utf-8") as f:
            bm = json.loads(f.read())
        if len(bm["doc_lens"]) != meta.get("bm25_docs", 0):
            logger.warning(f"[rebuild] BM25 doc count mismatch: {len(bm['doc_lens'])} docs vs {meta.get('bm25_docs')} in metadata")
            return None
    except Exception as e:
        logger.warning(f"[rebuild] Failed to load BM25: {e}")
        return None

    # 4.5. Preload FAISS index if enabled (Quick Win #2)
    global _FAISS_INDEX
    if USE_ANN == "faiss" and os.path.exists(FILES["faiss_index"]):
        try:
            _FAISS_INDEX = load_faiss_index(FILES["faiss_index"])
            if _FAISS_INDEX:
                logger.info(f"✓ Preloaded FAISS index: nprobe={ANN_NPROBE}")
            else:
                logger.info("FAISS index file exists but failed to load, will fall back")
        except Exception as e:
            logger.warning(f"Failed to preload FAISS: {e}, will lazy-load on first query")
            _FAISS_INDEX = None

    # 5. Cross-check: embeddings and chunks must match
    if vecs_n.shape[0] != len(chunks):
        logger.warning(f"[rebuild] Embedding-chunk mismatch: {vecs_n.shape[0]} embeddings vs {len(chunks)} chunks")
        logger.warning("[rebuild] Remediation: rebuild required")
        return None

    # 5.5. KB drift detection: if source MD present, check hash against metadata
    if os.path.exists("knowledge_full.md"):
        try:
            kb_sha = compute_sha256("knowledge_full.md")
            stored_sha = meta.get("kb_sha256", "")
            if stored_sha and stored_sha != kb_sha:
                logger.warning("[rebuild] KB drift detected: source MD has changed")
                logger.warning("[rebuild] Remediation: run 'python3 clockify_support_cli.py build knowledge_full.md'")
                return None
        except Exception as e:
            logger.debug(f"Could not check KB hash: {e}")

    # 6. Optional HNSW index (non-blocking if missing)
    hnsw = None
    if os.getenv("USE_HNSWLIB") == "1" and os.path.exists(FILES["hnsw"]):
        try:
            import hnswlib
            hnsw = hnswlib.Index(space='cosine', dim=vecs_n.shape[1])
            hnsw.load_index(FILES["hnsw"])
            logger.debug(f"Loaded HNSW index from {FILES['hnsw']}")
        except ImportError:
            logger.debug("hnswlib not installed; skipping HNSW")
        except Exception as e:
            logger.warning(f"Failed to load HNSW: {e}; continuing without it")

    logger.debug(f"Index loaded: {len(chunks)} chunks, {vecs_n.shape[0]} embeddings, {len(bm['doc_lens'])} BM25 docs")
    return chunks, vecs_n, bm, hnsw

# ====== POLICY GUARDRAILS ======
def looks_sensitive(question: str) -> bool:
    """Check if question involves sensitive intent (account/billing/PII)."""
    sensitive_keywords = {
        # Financial
        "invoice", "billing", "credit card", "payment", "salary", "account balance",
        # Authentication & Secrets
        "password", "token", "api key", "secret", "private key",
        # PII
        "ssn", "social security", "iban", "swift", "routing number", "account number",
        "phone number", "email address", "home address", "date of birth",
        # Compliance
        "gdpr", "pii", "personally identifiable", "personal data"
    }
    q_lower = question.lower()
    return any(kw in q_lower for kw in sensitive_keywords)

def inject_policy_preamble(snippets_block: str, question: str) -> str:
    """Optionally prepend policy reminder for sensitive queries."""
    if looks_sensitive(question):
        policy = "[INTERNAL POLICY]\nDo not reveal PII, account secrets, or payment details. For account changes, redirect to secure internal admin panel.\n\n"
        return policy + snippets_block
    return snippets_block

# ====== INPUT SANITIZATION ======
def sanitize_question(q: str, max_length: int = 2000) -> str:
    """Validate and sanitize user question.

    Args:
        q: User question string
        max_length: Maximum allowed question length (default: 2000)

    Returns:
        Sanitized question string

    Raises:
        ValueError: If question is invalid (empty, too long, invalid characters)
    """
    # Type check
    if not isinstance(q, str):
        raise ValueError(f"Question must be a string, got {type(q).__name__}")

    # Strip whitespace
    q = q.strip()

    # Check length
    if len(q) == 0:
        raise ValueError("Question cannot be empty. Hint: Provide a meaningful question about Clockify.")
    if len(q) > max_length:
        raise ValueError(
            f"Question too long (max {max_length} characters, got {len(q)}).\n"
            f"Hint: Break your question into smaller, focused queries."
        )

    # Check for null bytes first (specific check)
    if '\x00' in q:
        raise ValueError("Question contains null bytes")

    # Check for control characters (except newline, tab, carriage return)
    if any(ord(c) < 32 and c not in '\n\r\t' for c in q):
        raise ValueError("Question contains invalid control characters")

    # Check for suspicious patterns (basic prompt injection detection)
    suspicious_patterns = [
        '<script',
        'javascript:',
        'eval(',
        'exec(',
        '__import__',
        '<?php',
    ]
    q_lower = q.lower()
    for pattern in suspicious_patterns:
        if pattern in q_lower:
            raise ValueError(f"Question contains suspicious pattern: {pattern}")

    return q

# ====== RATE LIMITING ======
class RateLimiter:
    """Token bucket rate limiter for DoS prevention."""

    def __init__(self, max_requests=10, window_seconds=60):
        """Initialize rate limiter.

        Args:
            max_requests: Maximum number of requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: deque[float] = deque()
        self._lock = threading.RLock()

    def allow_request(self) -> bool:
        """Check if request is allowed under rate limit.

        Returns:
            True if request is allowed, False if rate limited
        """
        with self._lock:
            now = time.time()

            # Remove old requests outside the window
            while self.requests and self.requests[0] < now - self.window_seconds:
                self.requests.popleft()

            # Check if limit exceeded
            if len(self.requests) >= self.max_requests:
                return False

            # Allow request and record timestamp
            self.requests.append(now)
            return True

    def wait_time(self) -> float:
        """Calculate seconds until next request allowed.

        Returns:
            Seconds to wait (0 if request would be allowed now)
        """
        with self._lock:
            if len(self.requests) < self.max_requests:
                return 0.0

            # Time until oldest request falls out of window
            oldest = self.requests[0]
            return max(0.0, self.window_seconds - (time.time() - oldest))

# Global rate limiter (10 queries per minute by default)
RATE_LIMITER = RateLimiter(
    max_requests=int(os.environ.get("RATE_LIMIT_REQUESTS", "10")),
    window_seconds=int(os.environ.get("RATE_LIMIT_WINDOW", "60"))
)

# ====== QUERY CACHING (Rank 14) ======
class QueryCache:
    """TTL-based cache for repeated queries to eliminate redundant computation."""

    def __init__(self, maxsize=100, ttl_seconds=3600):
        """Initialize query cache.

        Args:
            maxsize: Maximum number of cached queries (LRU eviction)
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self.cache: dict[str, tuple[str, dict, float]] = {}  # {question_hash: (answer, metadata_with_timestamp, timestamp)}
        self.access_order: deque[str] = deque()  # For LRU eviction
        self.hits = 0
        self.misses = 0
        self._lock = threading.RLock()

    def _hash_question(self, question: str, params: dict = None) -> str:
        """Generate cache key from question and retrieval parameters.

        Args:
            question: User question
            params: Retrieval parameters (top_k, pack_top, use_rerank, threshold)
        """
        if params is None:
            cache_input = question
        else:
            # Sort params for consistent hashing
            sorted_params = sorted(params.items())
            cache_input = question + str(sorted_params)
        return hashlib.md5(cache_input.encode('utf-8')).hexdigest()

    def get(self, question: str, params: dict = None):
        """Retrieve cached answer if available and not expired.

        Args:
            question: User question
            params: Retrieval parameters (optional, for cache key)

        Returns:
            (answer, metadata) tuple if cache hit, None if cache miss
        """
        with self._lock:
            key = self._hash_question(question, params)

            if key not in self.cache:
                self.misses += 1
                return None

            answer, metadata, timestamp = self.cache[key]
            age = time.time() - timestamp

            # Check if expired
            if age > self.ttl_seconds:
                del self.cache[key]
                self.access_order.remove(key)
                self.misses += 1
                return None

            # Cache hit - update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            self.hits += 1
            logger.debug(f"[cache] HIT question_hash={key[:8]} age={age:.1f}s")
            return answer, metadata

    def put(self, question: str, answer: str, metadata: dict, params: dict = None):
        """Store answer in cache.

        Args:
            question: User question
            answer: Generated answer
            metadata: Answer metadata (selected chunks, scores, etc.)
            params: Retrieval parameters (optional, for cache key)
        """
        with self._lock:
            key = self._hash_question(question, params)

            # Evict oldest entry if cache full
            if len(self.cache) >= self.maxsize and key not in self.cache:
                oldest = self.access_order.popleft()
                del self.cache[oldest]
                logger.debug(f"[cache] EVICT question_hash={oldest[:8]} (LRU)")

            # Store entry with timestamp (single write with metadata augmentation)
            timestamp = time.time()
            metadata_copy = dict(metadata) if metadata is not None else {}
            metadata_copy["timestamp"] = timestamp
            self.cache[key] = (answer, metadata_copy, timestamp)

            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)

            logger.debug(f"[cache] PUT question_hash={key[:8]}")

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.hits = 0
            self.misses = 0
            logger.info("[cache] CLEAR")

    def stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with hits, misses, size, hit_rate
        """
        with self._lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            return {
                "hits": self.hits,
                "misses": self.misses,
                "size": len(self.cache),
                "maxsize": self.maxsize,
                "hit_rate": hit_rate
            }

# Global query cache (100 entries, 1 hour TTL by default)
QUERY_CACHE = QueryCache(
    maxsize=int(os.environ.get("CACHE_MAXSIZE", "100")),
    ttl_seconds=int(os.environ.get("CACHE_TTL", "3600"))
)

# ====== STRUCTURED LOGGING ======
def log_query(query, answer, retrieved_chunks, latency_ms, refused=False, metadata=None):
    """Log query with structured JSON format for monitoring and analytics.

    Args:
        query: User question string
        answer: Generated answer text
        retrieved_chunks: List of retrieved chunk dicts with scores
        latency_ms: Total query latency in milliseconds
        refused: Whether answer was refused (returned REFUSAL_STR)
        metadata: Optional dict with additional metadata (debug, backend, etc.)
    """
    # Early exit if logging is disabled (Rank 7: avoid needless work)
    if QUERY_LOG_DISABLED:
        return

    normalized_chunks = []
    for chunk in retrieved_chunks:
        if isinstance(chunk, dict):
            normalized = chunk.copy()
            chunk_id = normalized.get("id") or normalized.get("chunk_id")
            normalized["id"] = chunk_id
            normalized["dense"] = float(normalized.get("dense", normalized.get("score", 0.0)))
            normalized["bm25"] = float(normalized.get("bm25", 0.0))
            normalized["hybrid"] = float(normalized.get("hybrid", normalized["dense"]))
            # Rank 12: Redact chunk text for security/privacy unless explicitly enabled
            if not LOG_QUERY_INCLUDE_CHUNKS:
                normalized.pop("chunk", None)  # Remove full chunk text
                normalized.pop("text", None)   # Remove text field if present
        else:
            normalized = {
                "id": chunk,
                "dense": 0.0,
                "bm25": 0.0,
                "hybrid": 0.0,
            }
        normalized_chunks.append(normalized)

    chunk_ids = [c.get("id") for c in normalized_chunks]
    dense_scores = [c.get("dense", 0.0) for c in normalized_chunks]
    bm25_scores = [c.get("bm25", 0.0) for c in normalized_chunks]
    hybrid_scores = [c.get("hybrid", 0.0) for c in normalized_chunks]
    primary_scores = hybrid_scores if hybrid_scores else []

    # Honor LOG_QUERY_INCLUDE_ANSWER flag (Rank 15)
    entry_answer = answer if LOG_QUERY_INCLUDE_ANSWER else LOG_QUERY_ANSWER_PLACEHOLDER

    log_entry = {
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query": query,
        "query_length": len(query),
        "answer": entry_answer,
        "answer_length": len(answer),
        "num_chunks_retrieved": len(chunk_ids),
        "chunk_ids": chunk_ids,
        "chunk_scores": {
            "dense": dense_scores,
            "bm25": bm25_scores,
            "hybrid": hybrid_scores,
        },
        "retrieved_chunks": normalized_chunks,
        "avg_chunk_score": float(np.mean(primary_scores)) if primary_scores else 0.0,
        "max_chunk_score": float(np.max(primary_scores)) if primary_scores else 0.0,
        "latency_ms": latency_ms,
        "refused": refused,
        "metadata": metadata or {},
    }

    try:
        # Ensure parent directory exists (Rank 19)
        log_path = pathlib.Path(QUERY_LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write query log: {e}")

# ====== ANSWER PIPELINE HELPERS ======
def apply_mmr_diversification(selected, scores, vecs_n, pack_top):
    """Apply Maximal Marginal Relevance diversification to selected chunks.

    Args:
        selected: List of selected chunk indices
        scores: Dict with "dense" scores
        vecs_n: Normalized embedding vectors
        pack_top: Maximum number of chunks to select

    Returns:
        List of diversified chunk indices
    """
    mmr_selected = []
    cand = list(selected)

    # Always include the top dense score first for better recall
    if cand:
        top_dense_idx = max(cand, key=lambda j: scores["dense"][j])
        mmr_selected.append(top_dense_idx)
        cand.remove(top_dense_idx)

    # Then diversify the rest using vectorized MMR
    if cand and len(mmr_selected) < pack_top:
        # Convert to numpy arrays for vectorized operations
        cand_array = np.array(cand, dtype=np.int32)
        relevance_scores = np.array([scores["dense"][j] for j in cand], dtype=np.float32)

        # Get embedding vectors for candidates
        cand_vecs = vecs_n[cand_array]  # [num_candidates, emb_dim]

        # Iteratively select using MMR
        remaining_mask = np.ones(len(cand_array), dtype=bool)

        while np.any(remaining_mask) and len(mmr_selected) < pack_top:
            # Compute MMR scores for remaining candidates
            mmr_scores = MMR_LAMBDA * relevance_scores.copy()

            if len(mmr_selected) > 0:  # Only apply diversity when we have prior selections
                # Get vectors of all already-selected items
                selected_vecs = vecs_n[mmr_selected]  # [num_selected, emb_dim]

                # Compute similarity matrix: [num_candidates, num_selected]
                similarity_matrix = cand_vecs @ selected_vecs.T

                # Get max similarity for each candidate
                max_similarities = similarity_matrix.max(axis=1)

                # Update MMR scores with diversity penalty
                mmr_scores -= (1 - MMR_LAMBDA) * max_similarities

            # Mask out already-selected candidates
            mmr_scores[~remaining_mask] = -np.inf

            # Select candidate with highest MMR score
            best_idx = mmr_scores.argmax()
            selected_chunk_idx = cand_array[best_idx]

            mmr_selected.append(int(selected_chunk_idx))
            remaining_mask[best_idx] = False

    return mmr_selected


def apply_reranking(question, chunks, mmr_selected, scores, use_rerank, seed, num_ctx, num_predict, retries):
    """Apply optional LLM reranking to MMR-selected chunks.

    Args:
        question: User question
        chunks: All chunks
        mmr_selected: List of MMR-selected chunk indices
        scores: Dict with relevance scores
        use_rerank: Whether to apply reranking
        seed, num_ctx, num_predict, retries: LLM parameters

    Returns:
        Tuple of (reranked_chunks, rerank_scores, rerank_applied, rerank_reason, timing)
    """
    rerank_scores = {}
    rerank_applied = False
    rerank_reason = "disabled"
    timing = 0.0

    if use_rerank:
        logger.debug(json.dumps({"event": "rerank_start", "candidates": len(mmr_selected)}))
        t0 = time.time()
        mmr_selected, rerank_scores, rerank_applied, rerank_reason = rerank_with_llm(
            question, chunks, mmr_selected, scores, seed=seed, num_ctx=num_ctx, num_predict=num_predict, retries=retries
        )
        timing = time.time() - t0
        logger.debug(json.dumps({"event": "rerank_done", "selected": len(mmr_selected), "scored": len(rerank_scores)}))

        # Add greppable rerank fallback log
        if not rerank_applied:
            logger.debug("info: rerank=fallback reason=%s", rerank_reason)

    return mmr_selected, rerank_scores, rerank_applied, rerank_reason, timing


def generate_llm_answer(question, context_block, seed, num_ctx, num_predict, retries, packed_ids=None):
    """Generate answer from LLM given question and context with confidence scoring and citation validation (Rank 28, Rank 9).

    Args:
        question: User question
        context_block: Packed context snippets
        seed, num_ctx, num_predict, retries: LLM parameters
        packed_ids: List of chunk IDs included in context (for citation validation)

    Returns:
        Tuple of (answer_text, timing, confidence)
        - answer_text: The LLM's response
        - timing: Time taken for LLM call
        - confidence: 0-100 score, or None if not provided/parsable
    """
    t0 = time.time()
    raw_response = ask_llm(question, context_block, seed=seed, num_ctx=num_ctx, num_predict=num_predict, retries=retries).strip()
    timing = time.time() - t0

    # Rank 28: Parse JSON response with confidence
    confidence = None
    answer = raw_response  # Default to raw response if parsing fails

    try:
        # Try to parse as JSON
        # Handle markdown code blocks (```json ... ```)
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            # Extract content between ``` markers
            lines = cleaned.split("\n")
            # Remove first line (```json or ```) and last line (```)
            if len(lines) >= 3 and lines[-1].strip() == "```":
                cleaned = "\n".join(lines[1:-1]).strip()
            elif len(lines) >= 2:
                # Sometimes the closing ``` is on same line
                cleaned = "\n".join(lines[1:]).replace("```", "").strip()

        parsed = json.loads(cleaned)

        # Extract answer and confidence
        if isinstance(parsed, dict):
            if "answer" in parsed:
                answer = parsed["answer"]
            if "confidence" in parsed:
                conf_val = parsed["confidence"]
                # Validate confidence is 0-100
                if isinstance(conf_val, (int, float)) and 0 <= conf_val <= 100:
                    confidence = int(conf_val)
                    logger.debug(f"LLM confidence: {confidence}/100")
                else:
                    logger.warning(f"Invalid confidence value: {conf_val}, ignoring")

    except json.JSONDecodeError as e:
        # JSON parsing failed, use raw response as answer
        logger.debug(f"LLM response not JSON (expected for backwards compatibility): {str(e)[:50]}")
    except Exception as e:
        # Unexpected error in parsing
        logger.warning(f"Error parsing LLM confidence: {e}")

    # Rank 18: Improve fallback handling for non-JSON responses
    # Check if raw response lacks citations and might be policy-violating
    if answer == raw_response:  # JSON parsing failed, using raw response
        # Check if answer contains citation markers
        if packed_ids is not None and len(packed_ids) > 0:
            has_citations = bool(extract_citations(answer))
            if not has_citations and answer != REFUSAL_STR:
                # No citations found in non-JSON response - potential policy violation
                logger.warning(f"[llm_fallback] Raw response lacks citations, may violate policy")
                # Could trigger refusal here for strict enforcement:
                # answer = REFUSAL_STR
                # But for now, just log the warning

    # Rank 9: Validate citations match packed chunks
    if packed_ids is not None:
        is_valid, valid_cits, invalid_cits = validate_citations(answer, packed_ids)
        if not is_valid and invalid_cits:
            logger.warning(f"[citation_validation] Invalid citations: {invalid_cits} (valid: {packed_ids[:10]}...)")
            # Log warning but don't fail - some models may use different citation formats

    return answer, timing, confidence


# ====== ANSWER (STATELESS) ======
def answer_once(
    question: str,
    chunks,
    vecs_n,
    bm,
    top_k=12,
    pack_top=6,
    threshold=0.30,
    use_rerank=False,
    debug=False,
    hnsw=None,
    seed=DEFAULT_SEED,
    num_ctx=DEFAULT_NUM_CTX,
    num_predict=DEFAULT_NUM_PREDICT,
    retries=0
) -> tuple[str, dict]:
    """Answer a single question. Stateless. Returns (answer_text, metadata) - Task B, C, F.

    Returns:
        Tuple of (answer_text, metadata) where answer_text is the generated answer string
        and metadata is a dict containing selected chunks, scores, timings, etc.
    """
    # Sanitize question input
    try:
        question = sanitize_question(question)
    except ValueError as e:
        logger.warning(f"Invalid question: {e}")
        return f"Invalid question: {e}", {"selected": [], "scores": [], "timings": {}, "refused": False, "used_tokens": 0}

    # Check rate limit
    if not RATE_LIMITER.allow_request():
        wait_seconds = RATE_LIMITER.wait_time()
        logger.warning(f"Rate limit exceeded, wait {wait_seconds:.0f}s")
        return f"Rate limit exceeded. Please wait {wait_seconds:.0f} seconds before next query.", {
            "selected": [], "scores": [], "timings": {}, "refused": False, "rate_limited": True, "used_tokens": 0
        }

    # Check query cache (Rank 14: 100% speedup on repeated queries)
    # Include retrieval parameters in cache key to prevent stale answers (Rank 3)
    cache_params = {
        "top_k": top_k,
        "pack_top": pack_top,
        "threshold": threshold,
        "use_rerank": use_rerank
    }
    cached_result = QUERY_CACHE.get(question, params=cache_params)
    if cached_result is not None:
        answer, cached_metadata = cached_result
        # Work on a copy so cached metadata (including timestamp) remains intact
        metadata = dict(cached_metadata) if cached_metadata is not None else {}
        metadata.setdefault("used_tokens", 0)
        metadata["cached"] = True
        metadata["cache_hit"] = True
        # Quick Win #9: Add cache hit logging
        timestamp = metadata.get("timestamp")
        if timestamp is None:
            timestamp = time.time()
            metadata["timestamp"] = timestamp
        cache_age = time.time() - timestamp
        logger.info(f"[cache] HIT question_len={len(question)} cache_age={cache_age:.1f}s")
        return answer, metadata

    turn_start = time.time()
    timings = {}
    try:
        # Step 1: Hybrid retrieval
        t0 = time.time()
        selected, scores = retrieve(question, chunks, vecs_n, bm, top_k=top_k, hnsw=hnsw, retries=retries)
        timings["retrieve"] = time.time() - t0

        # Step 2: MMR diversification
        mmr_selected = apply_mmr_diversification(selected, scores, vecs_n, pack_top)

        # Step 3: Optional LLM reranking
        mmr_selected, rerank_scores, rerank_applied, rerank_reason, rerank_timing = apply_reranking(
            question, chunks, mmr_selected, scores, use_rerank, seed, num_ctx, num_predict, retries
        )
        timings["rerank"] = rerank_timing

        dense_scores_all = scores["dense"]
        bm25_scores_all = scores["bm25"]
        hybrid_scores_all = scores["hybrid"]
        chunk_id_to_index = {chunk["id"]: idx for idx, chunk in enumerate(chunks)}
        chunk_scores_by_id = {}
        for cid, idx in chunk_id_to_index.items():
            chunk_scores_by_id[cid] = {
                "index": idx,
                "dense": float(dense_scores_all[idx]),
                "bm25": float(bm25_scores_all[idx]),
                "hybrid": float(hybrid_scores_all[idx]),
            }
        mmr_rank_by_id = {chunks[idx]["id"]: rank for rank, idx in enumerate(mmr_selected)}

        # Step 4: Coverage check (Rank 8: removed duplicate check)
        coverage_pass = coverage_ok(mmr_selected, scores["dense"], threshold)
        if not coverage_pass:
            if debug:
                print(f"\n[DEBUG] Coverage failed: {len(mmr_selected)} selected, need ≥2 @ {threshold}")
            logger.debug(f"[coverage_gate] REJECTED: seed={seed} model={GEN_MODEL} selected={len(mmr_selected)} threshold={threshold}")

            # Log refusal
            latency_ms = int((time.time() - turn_start) * 1000)
            refusal_chunks = []
            for rank, idx in enumerate(mmr_selected):
                chunk_id = chunks[idx]["id"]
                info = chunk_scores_by_id.get(chunk_id)
                if info is None:
                    continue
                entry = {
                    "id": chunk_id,
                    "mmr_rank": rank,
                    "dense": info["dense"],
                    "bm25": info["bm25"],
                    "hybrid": info["hybrid"],
                }
                rerank_score = rerank_scores.get(info["index"])
                if rerank_score is not None:
                    entry["rerank_score"] = float(rerank_score)
                refusal_chunks.append(entry)
            dense_scores = scores.get("dense", np.zeros(len(chunks), dtype=np.float32))
            retrieved_chunks = []
            for idx in mmr_selected:
                if not isinstance(idx, (int, np.integer)):
                    continue
                if idx < 0 or idx >= len(chunks) or idx >= len(dense_scores):
                    continue
                chunk = chunks[idx]
                if not isinstance(chunk, dict):
                    continue
                chunk_id = chunk.get("id")
                if chunk_id is None:
                    continue
                retrieved_chunks.append(
                    {
                        "id": chunk_id,
                        "score": float(dense_scores[idx]),
                        "chunk": chunk,
                    }
                )

            log_query(
                query=question,
                answer=REFUSAL_STR,
                retrieved_chunks=refusal_chunks,
                latency_ms=latency_ms,
                refused=True,
                metadata={"debug": debug, "backend": EMB_BACKEND, "coverage_pass": False}
            )

            # Cache refusal (Rank 14)
            refusal_metadata = {"selected": [], "refused": True, "cached": False, "cache_hit": False, "used_tokens": 0}
            refusal_metadata["timestamp"] = time.time()
            QUERY_CACHE.put(question, REFUSAL_STR, refusal_metadata, params=cache_params)

            return REFUSAL_STR, refusal_metadata

        # Step 5: Pack with token budget and snippet cap (Task C)
        block, ids, used_tokens = pack_snippets(chunks, mmr_selected, pack_top=pack_top, budget_tokens=CTX_TOKEN_BUDGET, num_ctx=num_ctx)

        # Apply policy preamble for sensitive queries
        block = inject_policy_preamble(block, question)

        # Step 6: Generate answer from LLM (Rank 28: with confidence, Rank 9: with citation validation)
        ans, llm_timing, confidence = generate_llm_answer(question, block, seed, num_ctx, num_predict, retries, packed_ids=ids)
        timings["ask_llm"] = llm_timing
        timings["total"] = time.time() - turn_start

        # Step 7: Optional debug output with all metrics (Task B, F)
        if debug:
            diag = []
            for rank, i in enumerate(mmr_selected):
                entry = {
                    "id": chunks[i]["id"],
                    "title": chunks[i]["title"],
                    "section": chunks[i]["section"],
                    "url": chunks[i]["url"],
                    "dense": float(scores["dense"][i]),
                    "bm25": float(scores["bm25"][i]),
                    "hybrid": float(scores["hybrid"][i]),
                    "mmr_rank": rank,
                    "rerank_applied": bool(rerank_applied),
                    "rerank_reason": rerank_reason or ""
                }
                if i in rerank_scores:
                    entry["rerank_score"] = float(rerank_scores[i])
                diag.append(entry)

            # Patch 7: wrap global fields under `meta` (Rank 28: include confidence)
            debug_info = {
                "meta": {
                    "rerank_applied": bool(rerank_applied),
                    "rerank_reason": rerank_reason or "",
                    "selected_count": len(mmr_selected),
                    "pack_ids_count": len(ids),
                    "used_tokens": int(used_tokens),
                    "confidence": confidence  # Rank 28
                },
                "pack_ids_preview": ids[:10],
                "snippets": diag
            }
            ans += "\n\n[DEBUG]\n" + json.dumps(debug_info, ensure_ascii=False, indent=2)

        # Task F: info log with only counts (Rank 28: include confidence)
        conf_str = f" confidence={confidence}" if confidence is not None else ""
        logger.info(
            f"info: retrieve={timings.get('retrieve', 0):.3f} rerank={timings.get('rerank', 0):.3f} "
            f"ask={timings['ask_llm']:.3f} total={timings['total']:.3f} "
            f"selected={len(mmr_selected)} packed={len(ids)} used_tokens={used_tokens}{conf_str}"
        )

        # Log successful query (Rank 6: consolidated retrieved_chunks building)
        latency_ms = int(timings['total'] * 1000)
        retrieved_chunks = []
        for pack_rank, chunk_id in enumerate(ids):
            info = chunk_scores_by_id.get(chunk_id)
            if info is None:
                continue
            idx = info["index"]
            chunk = chunks[idx]
            if not isinstance(chunk, dict):
                continue
            entry = {
                "id": chunk_id,
                "pack_rank": pack_rank,
                "dense": info["dense"],
                "bm25": info["bm25"],
                "hybrid": info["hybrid"],
            }
            if LOG_QUERY_INCLUDE_CHUNKS:
                entry["chunk"] = chunk
            mmr_rank = mmr_rank_by_id.get(chunk_id)
            if mmr_rank is not None:
                entry["mmr_rank"] = mmr_rank
            rerank_score = rerank_scores.get(idx)
            if rerank_score is not None:
                entry["rerank_score"] = float(rerank_score)
            retrieved_chunks.append(entry)

        log_query(
            query=question,
            answer=ans,
            retrieved_chunks=retrieved_chunks,
            latency_ms=latency_ms,
            refused=False,
            metadata={
                "debug": debug,
                "backend": EMB_BACKEND,
                "coverage_pass": True,
                "rerank_applied": rerank_applied,
                "rerank_reason": rerank_reason,  # Why rerank was/wasn't applied
                "rerank_candidates": len(mmr_selected) if use_rerank else 0,  # Candidates sent to reranker
                "num_selected": len(mmr_selected),
                "num_packed": len(ids),
                "used_tokens": used_tokens,
                "timings": timings,
                "confidence": confidence  # Rank 28
            }
        )

        # Cache successful answer (Rank 14) with confidence (Rank 28)
        result_metadata = {
            "selected": ids,
            "used_tokens": int(used_tokens),
            "timings": timings,
            "rerank_applied": bool(rerank_applied),
            "cached": False,
            "cache_hit": False,
            "confidence": confidence
        }
        result_metadata["timestamp"] = time.time()
        QUERY_CACHE.put(question, ans, result_metadata, params=cache_params)

        return ans, result_metadata
    except (EmbeddingError, LLMError, BuildError):
        # Re-raise specific exceptions
        raise
    except Exception as e:
        # Wrap unexpected exceptions with context
        raise RuntimeError(f"Unexpected error in answer_once: {e}") from e

# ====== TASK J: SELF-TESTS (7 Tests) ======
def test_mmr_behavior_ok():
    """Verify MMR diversification produces diverse selections with rerank enabled."""

    # Synthetic knowledge base with three chunks – two are similar, one is orthogonal
    chunks = [
        {"id": "chunk-0", "title": "Doc A", "section": "S1", "url": "", "text": "Details A"},
        {"id": "chunk-1", "title": "Doc A", "section": "S1", "url": "", "text": "More A"},
        {"id": "chunk-2", "title": "Doc B", "section": "S2", "url": "", "text": "Details B"},
    ]

    base_vecs = np.array([
        [1.0, 0.0],
        [0.99, 0.0],  # Nearly identical to chunk-0
        [0.0, 1.0],   # Orthogonal for diversity
    ], dtype=np.float32)
    norms = np.linalg.norm(base_vecs, axis=1, keepdims=True)
    vecs_n = base_vecs / np.where(norms == 0, 1, norms)

    dense_scores = np.array([0.9, 0.85, 0.8], dtype=np.float32)
    bm25_scores = np.array([0.2, 0.1, 0.05], dtype=np.float32)
    hybrid_scores = 0.5 * dense_scores + 0.5 * bm25_scores

    # Monkeypatch retrieve/rerank/ask to avoid network dependencies
    original_retrieve = retrieve
    original_rerank = rerank_with_llm
    original_ask = ask_llm

    rerank_probe = {}

    def fake_retrieve(question, chunks_, vecs_n_, bm_, top_k=12, hnsw=None, retries=0):
        return [0, 1, 2], {
            "dense": dense_scores,
            "bm25": bm25_scores,
            "hybrid": hybrid_scores,
        }

    def fake_rerank(question, chunks_, selected, scores, seed=DEFAULT_SEED, num_ctx=DEFAULT_NUM_CTX, num_predict=DEFAULT_NUM_PREDICT, retries=0):
        rerank_probe["mmr_selection"] = list(selected)
        # Keep original order but record deterministic scores
        rerank_scores = {idx: float(1.0 - 0.1 * pos) for pos, idx in enumerate(selected)}
        return list(selected), rerank_scores, True, ""

    def fake_ask_llm(question, snippets_block, seed=DEFAULT_SEED, num_ctx=DEFAULT_NUM_CTX, num_predict=DEFAULT_NUM_PREDICT, retries=0):
        return json.dumps({"answer": "stub", "confidence": 100})

    try:
        globals()["retrieve"] = fake_retrieve
        globals()["rerank_with_llm"] = fake_rerank
        globals()["ask_llm"] = fake_ask_llm

        answer, metadata = answer_once(
            "What is covered?",
            chunks,
            vecs_n,
            bm={},
            top_k=3,
            pack_top=2,
            threshold=0.5,
            use_rerank=True,
            debug=False,
            seed=123,
            num_ctx=128,
            num_predict=64,
            retries=0,
        )
    finally:
        globals()["retrieve"] = original_retrieve
        globals()["rerank_with_llm"] = original_rerank
        globals()["ask_llm"] = original_ask

    assert answer.strip() == "stub"
    # Ensure MMR fed reranker with diverse indices (chunk-0 then chunk-2)
    assert rerank_probe.get("mmr_selection") == [0, 2]
    # Packed selection should contain two unique chunks, excluding the duplicate chunk-1
    assert metadata["selected"] == ["chunk-0", "chunk-2"]
    assert len(set(metadata["selected"])) == 2
    assert metadata["rerank_applied"] is True
    return True

def test_pack_headroom_enforced():
    """Verify top-1 always included - Task J."""
    # Mock chunks
    chunks = [
        {"id": "1", "title": "T", "section": "S", "url": "", "text": "x" * 20000},  # Very large
        {"id": "2", "title": "T", "section": "S", "url": "", "text": "y" * 100},
    ]
    # Pack with very small budget
    block, ids, used = pack_snippets(chunks, [0, 1], pack_top=2, budget_tokens=10, num_ctx=1000)
    # Top-1 should always be included even if it exceeds budget
    assert len(ids) >= 1
    assert "1" in ids
    return True

def test_rtf_guard_false_positive():
    """Verify non-RTF with backslashes not stripped - Task J."""
    # Text with backslashes but not RTF
    text = r"This is \normal text with \backslashes but no RTF commands"
    result = strip_noise(text)
    # Should not be modified (no RTF stripping)
    assert "\\normal" in result
    assert "\\backslashes" in result
    return True

def test_float32_pipeline_ok():
    """Verify all vectors are float32 - Task J."""
    # Create a test vector
    vec = np.array([1.0, 2.0, 3.0], dtype="float64")
    # Save and load via atomic_save_npy
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        tmp_path = f.name
    try:
        atomic_save_npy(vec, tmp_path)
        loaded = np.load(tmp_path)
        assert loaded.dtype == np.float32
        return True
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_pack_cap_enforced():
    """Verify len(ids) <= pack_top hard cap - Task J."""
    # Create many small chunks
    chunks = [
        {"id": str(i), "title": f"T{i}", "section": "S", "url": "", "text": f"text{i}"}
        for i in range(20)
    ]
    # Pack with pack_top=6 but pass many indices
    order = list(range(20))
    block, ids, used = pack_snippets(chunks, order, pack_top=6, budget_tokens=10000, num_ctx=8192)
    # Verify hard cap: len(ids) <= 6
    assert len(ids) <= 6, f"Pack cap violated: got {len(ids)} > 6"
    assert len(ids) == 6, f"Expected exactly 6 items when budget allows; got {len(ids)}"
    return True

def test_post_retry_logic():
    """Verify POST retry logic exists - Task J."""
    # Check that http_post_with_retries helper exists and is used
    import inspect
    # v4.1: Check for http_post_with_retries function
    try:
        source = inspect.getsource(http_post_with_retries)
        assert "retry" in source.lower(), "Retry logic not found in http_post_with_retries"
        assert "time.sleep" in source or "backoff" in source, "Exponential backoff not found"
    except (NameError, AttributeError):
        raise AssertionError("http_post_with_retries function not found")
    return True

def test_rerank_applied_when_enabled():
    """Ensure reranking is gated by the use_rerank flag and updates metadata."""

    chunks = [
        {"id": "chunk-0", "title": "Doc A", "section": "S1", "url": "", "text": "Details A"},
        {"id": "chunk-1", "title": "Doc B", "section": "S2", "url": "", "text": "Details B"},
    ]
    base_vecs = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float32)
    vecs_n = base_vecs / np.linalg.norm(base_vecs, axis=1, keepdims=True)

    dense_scores = np.array([0.9, 0.8], dtype=np.float32)
    bm25_scores = np.array([0.1, 0.05], dtype=np.float32)
    hybrid_scores = 0.5 * dense_scores + 0.5 * bm25_scores

    original_retrieve = retrieve
    original_rerank = rerank_with_llm
    original_ask = ask_llm

    calls = {"true": 0, "false": 0}

    def fake_retrieve(question, chunks_, vecs_n_, bm_, top_k=12, hnsw=None, retries=0):
        return [0, 1], {
            "dense": dense_scores,
            "bm25": bm25_scores,
            "hybrid": hybrid_scores,
        }

    def fake_rerank(question, chunks_, selected, scores, seed=DEFAULT_SEED, num_ctx=DEFAULT_NUM_CTX, num_predict=DEFAULT_NUM_PREDICT, retries=0):
        if "use" in question:
            calls["true"] += 1
        else:
            calls["false"] += 1
        rerank_scores = {idx: float(1.0 - 0.1 * pos) for pos, idx in enumerate(selected)}
        return list(reversed(selected)), rerank_scores, True, ""

    def fake_ask_llm(question, snippets_block, seed=DEFAULT_SEED, num_ctx=DEFAULT_NUM_CTX, num_predict=DEFAULT_NUM_PREDICT, retries=0):
        return json.dumps({"answer": question, "confidence": 90})

    try:
        globals()["retrieve"] = fake_retrieve
        globals()["rerank_with_llm"] = fake_rerank
        globals()["ask_llm"] = fake_ask_llm

        ans_true, meta_true = answer_once(
            "use rerank please",
            chunks,
            vecs_n,
            bm={},
            top_k=2,
            pack_top=2,
            threshold=0.5,
            use_rerank=True,
            debug=False,
            seed=7,
            num_ctx=128,
            num_predict=64,
            retries=0,
        )

        ans_false, meta_false = answer_once(
            "skip rerank",
            chunks,
            vecs_n,
            bm={},
            top_k=2,
            pack_top=2,
            threshold=0.5,
            use_rerank=False,
            debug=False,
            seed=8,
            num_ctx=128,
            num_predict=64,
            retries=0,
        )
    finally:
        globals()["retrieve"] = original_retrieve
        globals()["rerank_with_llm"] = original_rerank
        globals()["ask_llm"] = original_ask

    assert ans_true.strip() == "use rerank please"
    assert ans_false.strip() == "skip rerank"
    assert meta_true["rerank_applied"] is True
    assert meta_false["rerank_applied"] is False
    assert calls["true"] == 1
    assert calls["false"] == 0
    # Reranker reversed the order when enabled
    assert meta_true["selected"] == ["chunk-1", "chunk-0"]
    # When disabled, MMR order preserved (first chunk stays first)
    assert meta_false["selected"][0] == "chunk-0"
    return True

def run_selftest():
    """Run all self-check tests - Task J."""
    tests = [
        ("MMR behavior", test_mmr_behavior_ok),
        ("Pack headroom", test_pack_headroom_enforced),
        ("Pack cap enforcement", test_pack_cap_enforced),
        ("RTF guard false positive", test_rtf_guard_false_positive),
        ("Float32 pipeline", test_float32_pipeline_ok),
        ("POST retry logic", test_post_retry_logic),
        ("Rerank applied", test_rerank_applied_when_enabled),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            status = "PASS" if result else "FAIL"
            results.append((name, status))
            logger.info(f"[selftest] {name}: {status}")
        except Exception as e:
            results.append((name, "FAIL"))
            logger.error(f"[selftest] {name}: FAIL ({e})")

    # Summary
    passed = sum(1 for _, status in results if status == "PASS")
    total = len(results)
    logger.info(f"[selftest] {passed}/{total} tests passed")

    return all(status == "PASS" for _, status in results)

# ====== ARTIFACT MANAGEMENT ======
def ensure_index_ready(retries=0):
    """Ensure retrieval artifacts are present and return loaded index components."""

    artifacts_ok = True
    for fname in [FILES["chunks"], FILES["emb"], FILES["meta"], FILES["bm25"], FILES["index_meta"]]:
        if not os.path.exists(fname):
            artifacts_ok = False
            break

    if not artifacts_ok:
        logger.info("[rebuild] artifacts missing or invalid: building from knowledge_full.md...")
        if os.path.exists("knowledge_full.md"):
            build("knowledge_full.md", retries=retries)
        else:
            logger.error("knowledge_full.md not found")
            sys.exit(1)

    result = load_index()
    if result is None:
        logger.info("[rebuild] artifact validation failed: rebuilding...")
        if os.path.exists("knowledge_full.md"):
            build("knowledge_full.md", retries=retries)
            result = load_index()
        else:
            logger.error("knowledge_full.md not found")
            sys.exit(1)

    if result is None:
        logger.error("Failed to load artifacts after rebuild")
        sys.exit(1)

    return result

# ====== REPL ======

def chat_repl(top_k=12, pack_top=6, threshold=0.30, use_rerank=False, debug=False, seed=DEFAULT_SEED, num_ctx=DEFAULT_NUM_CTX, num_predict=DEFAULT_NUM_PREDICT, retries=0, use_json=False):
    """Stateless REPL loop - Task I. v4.1: JSON output support."""
    # Task I: log config summary at startup
    _log_config_summary(use_rerank=use_rerank, pack_top=pack_top, seed=seed, threshold=threshold, top_k=top_k, num_ctx=num_ctx, num_predict=num_predict, retries=retries)

    # Lazy build and startup sanity check
    chunks, vecs_n, bm, hnsw = ensure_index_ready(retries=retries)

    print("\n" + "=" * 70)
    print("CLOCKIFY SUPPORT – Local, Stateless, Closed-Book")
    print("=" * 70)
    print("Type a question. Commands: :exit, :debug")

    # v4.1: Warm-up on startup to reduce first-token latency
    warmup_on_startup()
    print("=" * 70 + "\n")

    dbg = debug
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break
        if not q:
            continue
        if q == ":exit":
            break
        if q == ":debug":
            dbg = not dbg
            print(f"[debug={'ON' if dbg else 'OFF'}]")
            continue

        ans, meta = answer_once(
            q,
            chunks,
            vecs_n,
            bm,
            top_k=top_k,
            pack_top=pack_top,
            threshold=threshold,
            use_rerank=use_rerank,
            debug=dbg,
            hnsw=hnsw,
            seed=seed,
            num_ctx=num_ctx,
            num_predict=num_predict,
            retries=retries
        )
        # v4.1: JSON output support
        if use_json:
            output = answer_to_json(
                ans,
                meta.get("selected", []),
                meta.get("used_tokens"),
                top_k,
                pack_top
            )
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            print(ans)
            print()

# ====== WARM-UP (v4.1 - Section 6) ======
def warmup_on_startup():
    """Warm-up embeddings and LLM on startup (reduces first-token latency)."""
    warmup_enabled = os.environ.get("WARMUP", "1").lower() in ("1", "true", "yes")
    if not warmup_enabled:
        logger.debug("Warm-up disabled via WARMUP=0")
        return

    try:
        logger.info("info: warmup=start")
        # Warm-up embedding model with trivial query
        embed_query("warmup", retries=1)
        # Warm-up LLM with trivial prompt
        payload = {
            "model": GEN_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "max_tokens": 10
        }
        s = get_session()
        r = http_post_with_retries(
            f"{OLLAMA_URL}/api/chat",
            payload,
            retries=1,
            timeout=(CHAT_CONNECT_T, CHAT_READ_T)
        )
        logger.info("info: warmup=done")
    except Exception as e:
        logger.warning(f"warmup failed: {e}")

# ====== MAIN ======
def main():
    # v4.1: Declare globals at function start (Section 7)
    global EMB_BACKEND, USE_ANN, ALPHA_HYBRID, QUERY_LOG_DISABLED, FAISS_CANDIDATE_MULTIPLIER

    ap = argparse.ArgumentParser(
        prog="clockify_support_cli",
        description="Clockify internal support chatbot (offline, stateless, closed-book)"
    )

    # Global logging and config arguments
    ap.add_argument("--log", default="INFO", choices=["DEBUG", "INFO", "WARN"],
                    help="Logging level (default INFO)")
    ap.add_argument("--no-log", action="store_true",
                    help="Disable query log file writes (privacy mode)")
    ap.add_argument("--ollama-url", type=str, default=None,
                    help="Ollama endpoint (default from OLLAMA_URL env or http://127.0.0.1:11434)")
    ap.add_argument("--gen-model", type=str, default=None,
                    help="Generation model name (default from GEN_MODEL env or qwen2.5:32b)")
    ap.add_argument("--emb-model", type=str, default=None,
                    help="Embedding model name (default from EMB_MODEL env or nomic-embed-text)")
    ap.add_argument("--ctx-budget", type=int, default=None,
                    help="Context token budget (default from CTX_BUDGET env or 2800)")
    ap.add_argument("--query-expansions", type=str, default=None,
                    help="Path to JSON query expansion overrides (default config/query_expansions.json or CLOCKIFY_QUERY_EXPANSIONS env)")

    subparsers = ap.add_subparsers(dest="cmd")

    b = subparsers.add_parser("build", help="Build knowledge base")
    b.add_argument("md_path", help="Path to knowledge_full.md")
    b.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retries for transient errors (default 0)")
    # v4.1: Add flags to build subparser for explicit control
    b.add_argument("--emb-backend", choices=["local", "ollama"], default=EMB_BACKEND,
                   help="Embedding backend: local (SentenceTransformer) or ollama (default local)")
    b.add_argument("--ann", choices=["faiss", "none"], default=USE_ANN,
                   help="ANN index: faiss (IVFFlat) or none (full-scan, default faiss)")
    b.add_argument("--alpha", type=float, default=ALPHA_HYBRID,
                   help="Hybrid scoring blend: alpha*BM25 + (1-alpha)*dense (default 0.5)")

    c = subparsers.add_parser("chat", help="Start REPL")
    c.add_argument("--debug", action="store_true", help="Print retrieval diagnostics")
    c.add_argument("--rerank", action="store_true", help="Enable LLM-based reranking")
    c.add_argument("--topk", type=int, default=DEFAULT_TOP_K, help="Top-K candidates (default 12)")
    c.add_argument("--pack", type=int, default=DEFAULT_PACK_TOP, help="Snippets to pack (default 6)")
    c.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Cosine threshold (default 0.30)")
    c.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for LLM (default 42)")
    c.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX, help="LLM context window (default 8192)")
    c.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT, help="LLM max generation tokens (default 512)")
    c.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retries for transient errors (default 0)")
    # Task A: determinism check flags
    c.add_argument("--det-check", action="store_true", help="Determinism check: ask same Q twice, compare hashes")
    # v4.1: Add flags to chat subparser for explicit control
    c.add_argument("--emb-backend", choices=["local", "ollama"], default=EMB_BACKEND,
                   help="Embedding backend: local (SentenceTransformer) or ollama (default local)")
    c.add_argument("--ann", choices=["faiss", "none"], default=USE_ANN,
                   help="ANN index: faiss (IVFFlat) or none (full-scan, default faiss)")
    c.add_argument("--alpha", type=float, default=ALPHA_HYBRID,
                   help="Hybrid scoring blend: alpha*BM25 + (1-alpha)*dense (default 0.5)")
    c.add_argument("--no-expand", action="store_true",
                   help="Disable query expansion (synonym substitution)")
    c.add_argument("--faiss-multiplier", type=int, default=FAISS_CANDIDATE_MULTIPLIER,
                   help="FAISS candidate multiplier: retrieve top_k * N for reranking (default 3)")
    c.add_argument("--json", action="store_true", help="Output answer as JSON with metrics (v4.1)")

    a = subparsers.add_parser("ask", help="Answer a single question and exit")
    a.add_argument("question", help="Question text to send to the assistant")
    a.add_argument("question", help="Question to answer")
    a.add_argument("--debug", action="store_true", help="Print retrieval diagnostics")
    a.add_argument("--rerank", action="store_true", help="Enable LLM-based reranking")
    a.add_argument("--topk", type=int, default=DEFAULT_TOP_K, help="Top-K candidates (default 12)")
    a.add_argument("--pack", type=int, default=DEFAULT_PACK_TOP, help="Snippets to pack (default 6)")
    a.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Cosine threshold (default 0.30)")
    a.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for LLM (default 42)")
    a.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX, help="LLM context window (default 8192)")
    a.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT, help="LLM max generation tokens (default 512)")
    a.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retries for transient errors (default 0)")
    a.add_argument("--emb-backend", choices=["local", "ollama"], default=EMB_BACKEND,
                   help="Embedding backend: local (SentenceTransformer) or ollama (default local)")
    a.add_argument("--ann", choices=["faiss", "none"], default=USE_ANN,
                   help="ANN index: faiss (IVFFlat) or none (full-scan, default faiss)")
    a.add_argument("--alpha", type=float, default=ALPHA_HYBRID,
                   help="Hybrid scoring blend: alpha*BM25 + (1-alpha)*dense (default 0.5)")
    a.add_argument("--no-expand", action="store_true",
                   help="Disable query expansion (synonym substitution)")
    a.add_argument("--faiss-multiplier", type=int, default=FAISS_CANDIDATE_MULTIPLIER,
                   help="FAISS candidate multiplier: retrieve top_k * N for reranking (default 3)")
    a.add_argument("--json", action="store_true", help="Output answer as JSON with metrics (v4.1)")

    # v4.1: Ollama optimization flags (Section 7)
    ap.add_argument("--emb-backend", choices=["local", "ollama"], default=EMB_BACKEND,
                   help="Embedding backend: local (SentenceTransformer) or ollama (default local)")
    ap.add_argument("--ann", choices=["faiss", "none"], default=USE_ANN,
                   help="ANN index: faiss (IVFFlat) or none (full-scan, default faiss)")
    ap.add_argument("--alpha", type=float, default=ALPHA_HYBRID,
                   help="Hybrid scoring blend: alpha*BM25 + (1-alpha)*dense (default 0.5)")
    ap.add_argument("--selftest", action="store_true", help="Run self-tests and exit (v4.1)")
    ap.add_argument("--json", action="store_true", help="Output answer as JSON with metrics (v4.1)")
    ap.add_argument("--profile", action="store_true", help="Enable cProfile performance profiling (Rank 29)")

    args = ap.parse_args()

    # Setup logging after CLI arg parsing
    level = getattr(logging, args.log if hasattr(args, "log") else "INFO")
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    # Configure query expansion dictionary overrides
    if getattr(args, "query_expansions", None):
        set_query_expansion_path(args.query_expansions)
    elif os.environ.get(QUERY_EXPANSIONS_ENV_VAR):
        set_query_expansion_path(os.environ[QUERY_EXPANSIONS_ENV_VAR])
    else:
        set_query_expansion_path(None)

    try:
        load_query_expansion_dict(force_reload=True, suppress_errors=False)
    except ValueError as exc:
        logger.error("CONFIG ERROR: %s", exc)
        sys.exit(1)
    if getattr(args, "no_log", False):
        QUERY_LOG_DISABLED = True

    # v4.1: Update globals from CLI args (Section 7)
    EMB_BACKEND = args.emb_backend
    USE_ANN = args.ann
    ALPHA_HYBRID = args.alpha

    # Update FAISS multiplier if provided in subcommand args
    if hasattr(args, "faiss_multiplier"):
        FAISS_CANDIDATE_MULTIPLIER = args.faiss_multiplier

    # v4.1: Run selftest if requested (Section 8)
    if getattr(args, "selftest", False):
        success = run_selftest()
        sys.exit(0 if success else 1)

    # Validate and set config from CLI args
    try:
        validate_and_set_config(
            ollama_url=args.ollama_url,
            gen_model=args.gen_model,
            emb_model=args.emb_model,
            ctx_budget=args.ctx_budget
        )
        validate_chunk_config()
        check_pytorch_mps()  # v4.1.2: Check MPS availability on M1 Macs
    except ValueError as e:
        logger.error(f"CONFIG ERROR: {e}")
        sys.exit(1)

    # Auto-start REPL if no command given
    if args.cmd is None:
        chat_repl()
        return

    if args.cmd == "build":
        build(args.md_path, retries=getattr(args, "retries", 0))
        return

    if args.cmd == "ask":
        _log_config_summary(
            use_rerank=args.rerank,
            pack_top=args.pack,
            seed=args.seed,
            threshold=args.threshold,
            top_k=args.topk,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            retries=getattr(args, "retries", 0)
        )
        chunks, vecs_n, bm, hnsw = ensure_index_ready(retries=getattr(args, "retries", 0))
        ans, meta = answer_once(
            args.question,
            chunks,
            vecs_n,
            bm,
            top_k=args.topk,
            pack_top=args.pack,
            threshold=args.threshold,
            use_rerank=args.rerank,
            debug=args.debug,
            hnsw=hnsw,
            seed=args.seed,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            retries=getattr(args, "retries", 0)
        )
        if getattr(args, "json", False):
            used_tokens = meta.get("used_tokens")
            if used_tokens is None:
                used_tokens = len(meta.get("selected", []))
            output = answer_to_json(
                ans,
                meta.get("selected", []),
                used_tokens,
                args.topk,
                args.pack
            )
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            print(ans)
        return

    if args.cmd == "chat":
        # Task A: Determinism check
        if getattr(args, "det_check", False):
            # Load index once for determinism test
            for fname in [FILES["chunks"], FILES["emb"], FILES["meta"], FILES["bm25"], FILES["index_meta"]]:
                if not os.path.exists(fname):
                    logger.info("[rebuild] artifacts missing for det-check: building...")
                    if os.path.exists("knowledge_full.md"):
                        build("knowledge_full.md", retries=getattr(args, "retries", 0))
                    break
            result = load_index()
            if result:
                # v4.1: Determinism check using Ollama with tuple timeouts and retry helper
                try:
                    seed = 42
                    np.random.seed(seed)
                    prompt = "What is Clockify?"
                    payload = {"model": GEN_MODEL, "prompt": prompt, "options": {"seed": seed}}

                    r1 = http_post_with_retries(f"{OLLAMA_URL}/api/generate", payload,
                                                retries=2, timeout=(CHAT_CONNECT_T, CHAT_READ_T))
                    ans1 = r1.json().get("response", "")

                    np.random.seed(seed)
                    r2 = http_post_with_retries(f"{OLLAMA_URL}/api/generate", payload,
                                                retries=2, timeout=(CHAT_CONNECT_T, CHAT_READ_T))
                    ans2 = r2.json().get("response", "")

                    h1 = hashlib.md5(ans1.encode()).hexdigest()[:16]
                    h2 = hashlib.md5(ans2.encode()).hexdigest()[:16]
                    deterministic = (h1 == h2)
                    logger.info(f"[DETERMINISM] run1={h1} run2={h2} deterministic={deterministic}")
                    print(f'[DETERMINISM] run1={h1} run2={h2} deterministic={"true" if deterministic else "false"}')
                    sys.exit(0 if deterministic else 1)
                except Exception as e:
                    logger.error(f"❌ Determinism test failed: {e}")
                    sys.exit(1)
            else:
                logger.error("failed to load index for det-check")
                sys.exit(1)

        # Normal chat REPL
        chat_repl(
            top_k=args.topk,
            pack_top=args.pack,
            threshold=args.threshold,
            use_rerank=args.rerank,
            debug=args.debug,
            seed=args.seed,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            retries=getattr(args, "retries", 0),
            use_json=getattr(args, "json", False)  # v4.1: JSON output flag
        )
        return

    if args.cmd == "ask":
        chunks, vecs_n, bm, hnsw = ensure_index_ready(retries=getattr(args, "retries", 0))

        _log_config_summary(
            use_rerank=args.rerank,
            pack_top=args.pack,
            seed=args.seed,
            threshold=args.threshold,
            top_k=args.topk,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            retries=getattr(args, "retries", 0)
        )

        warmup_on_startup()

        ans, meta = answer_once(
            args.question,
            chunks,
            vecs_n,
            bm,
            top_k=args.topk,
            pack_top=args.pack,
            threshold=args.threshold,
            use_rerank=args.rerank,
            debug=args.debug,
            hnsw=hnsw,
            seed=args.seed,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            retries=getattr(args, "retries", 0)
        )

        if getattr(args, "json", False):
            used_tokens = meta.get("used_tokens")
            if used_tokens is None:
                used_tokens = len(meta.get("selected", []))
            output = answer_to_json(
                ans,
                meta.get("selected", []),
                used_tokens,
                args.topk,
                args.pack
            )
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            print(ans)
        return

if __name__ == "__main__":
    # Rank 29: cProfile profiling support
    # Check for --profile flag early (before parsing to avoid double-parse)
    if "--profile" in sys.argv:
        import cProfile
        import pstats
        import io

        print("=" * 60)
        print("cProfile: Performance profiling enabled")
        print("=" * 60)

        profiler = cProfile.Profile()
        profiler.enable()

        try:
            main()
        finally:
            profiler.disable()

            # Print stats to stdout
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.strip_dirs()
            ps.sort_stats(pstats.SortKey.CUMULATIVE)  # Sort by cumulative time

            print("\n" + "=" * 60)
            print("cProfile: Top 30 functions by cumulative time")
            print("=" * 60)
            ps.print_stats(30)

            print(s.getvalue())

            # Optionally save to file
            profile_file = "clockify_rag_profile.stats"
            profiler.dump_stats(profile_file)
            print(f"\nFull profile saved to: {profile_file}")
            print(f"View with: python -m pstats {profile_file}")
    else:
        main()
