"""Configuration constants for Clockify RAG system."""

import os


# ====== OLLAMA CONFIG ======
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
GEN_MODEL = os.environ.get("GEN_MODEL", "qwen2.5:32b")
EMB_MODEL = os.environ.get("EMB_MODEL", "nomic-embed-text")

# ====== CHUNKING CONFIG ======
CHUNK_CHARS = 1600
CHUNK_OVERLAP = 200

# ====== RETRIEVAL CONFIG ======
DEFAULT_TOP_K = 12
DEFAULT_PACK_TOP = 6
DEFAULT_THRESHOLD = 0.30
DEFAULT_SEED = 42

# ====== BM25 CONFIG ======
# BM25 parameters (tuned for technical documentation)
# Lower k1 (1.2→1.0): Reduces term frequency saturation for repeated technical terms
# Lower b (0.75→0.65): Reduces length normalization penalty for longer docs
BM25_K1 = float(os.environ.get("BM25_K1", "1.0"))
BM25_B = float(os.environ.get("BM25_B", "0.65"))

# ====== LLM CONFIG ======
DEFAULT_NUM_CTX = 8192
DEFAULT_NUM_PREDICT = 512
DEFAULT_RETRIES = 0

# ====== MMR & CONTEXT BUDGET ======
MMR_LAMBDA = 0.7
CTX_TOKEN_BUDGET = int(os.environ.get("CTX_BUDGET", "2800"))  # ~11,200 chars, overridable

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
    """Global KPI tracking for performance metrics."""
    retrieve_ms = 0
    ann_ms = 0
    rerank_ms = 0
    ask_ms = 0


# ====== TIMEOUT CONFIG ======
# Task G: Deterministic timeouts (environment-configurable for ops)
EMB_CONNECT_T = float(os.environ.get("EMB_CONNECT_TIMEOUT", "3"))
EMB_READ_T = float(os.environ.get("EMB_READ_TIMEOUT", "60"))
CHAT_CONNECT_T = float(os.environ.get("CHAT_CONNECT_TIMEOUT", "3"))
CHAT_READ_T = float(os.environ.get("CHAT_READ_TIMEOUT", "120"))
RERANK_READ_T = float(os.environ.get("RERANK_READ_TIMEOUT", "180"))

# ====== EMBEDDING BATCHING CONFIG (Rank 10) ======
# Parallel embedding generation for faster KB builds (3-5x speedup)
EMB_MAX_WORKERS = int(os.environ.get("EMB_MAX_WORKERS", "8"))  # Concurrent requests
EMB_BATCH_SIZE = int(os.environ.get("EMB_BATCH_SIZE", "32"))  # Texts per batch

# ====== REFUSAL STRING ======
# Exact refusal string (ASCII quotes only)
REFUSAL_STR = "I don't know based on the MD."

# ====== LOGGING CONFIG ======


def _get_bool_env(var_name: str, default: str = "1") -> bool:
    """Read a boolean environment variable."""

    value = os.environ.get(var_name, default)
    return value.lower() not in {"0", "false", "no", "off", ""}


# Query logging configuration
QUERY_LOG_FILE = os.environ.get("RAG_LOG_FILE", "rag_queries.jsonl")
LOG_QUERY_INCLUDE_ANSWER = _get_bool_env("RAG_LOG_INCLUDE_ANSWER", "1")
LOG_QUERY_ANSWER_PLACEHOLDER = os.environ.get("RAG_LOG_ANSWER_PLACEHOLDER", "[REDACTED]")
LOG_QUERY_INCLUDE_CHUNKS = _get_bool_env("RAG_LOG_INCLUDE_CHUNKS", "0")  # Redact chunk text by default for security/privacy

# ====== FILE PATHS ======
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

# ====== BUILD LOCK CONFIG ======
BUILD_LOCK = ".build.lock"
BUILD_LOCK_TTL_SEC = int(os.environ.get("BUILD_LOCK_TTL_SEC", "900"))  # Task D: 15 minutes default

# ====== RETRIEVAL CONFIG (CONTINUED) ======
# FAISS/HNSW candidate generation (Quick Win #6)
FAISS_CANDIDATE_MULTIPLIER = 3  # Retrieve top_k * 3 candidates for reranking
ANN_CANDIDATE_MIN = 200  # Minimum candidates even if top_k is small

# Reranking (Quick Win #6)
RERANK_SNIPPET_MAX_CHARS = 500  # Truncate chunk text for reranking prompt
RERANK_MAX_CHUNKS = 12  # Maximum chunks to send to reranking

# Retrieval thresholds (Quick Win #6)
COVERAGE_MIN_CHUNKS = 2  # Minimum chunks above threshold to proceed
