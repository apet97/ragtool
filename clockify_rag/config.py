"""Configuration constants for Clockify RAG system."""

import logging
import os
from dataclasses import dataclass
from typing import Iterable, Optional

# FIX (Error #13): Helper functions for safe environment variable parsing
_logger = logging.getLogger(__name__)


def _get_env_value(
    primary: str,
    default: Optional[str] = None,
    legacy_keys: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """Read environment variables with optional legacy fallbacks.

    Args:
        primary: Preferred environment variable name (new `RAG_*` namespace)
        default: Default value if nothing is set
        legacy_keys: Older env var names to support for backwards compatibility

    Returns:
        The first non-empty environment value, or the provided default.
    """
    keys = [primary]
    if legacy_keys:
        keys.extend(legacy_keys)
    for key in keys:
        value = os.environ.get(key)
        if value is not None and value.strip() != "":
            return value.strip()
    return default


def _parse_env_float(key: str, default: float, min_val: float = None, max_val: float = None) -> float:
    """Parse float from environment with validation.

    FIX (Error #13): Prevents crashes from invalid env var values.

    Args:
        key: Environment variable name
        default: Default value if not set or invalid
        min_val: Minimum allowed value (optional)
        max_val: Maximum allowed value (optional)

    Returns:
        Parsed and validated float value
    """
    value = os.environ.get(key)
    if value is None:
        return default

    try:
        parsed = float(value)
    except ValueError as e:
        _logger.error(
            f"Invalid float for {key}='{value}': {e}. "
            f"Using default: {default}"
        )
        return default

    if min_val is not None and parsed < min_val:
        _logger.warning(f"{key}={parsed} below minimum {min_val}, clamping")
        return min_val
    if max_val is not None and parsed > max_val:
        _logger.warning(f"{key}={parsed} above maximum {max_val}, clamping")
        return max_val

    return parsed


def _parse_env_int(key: str, default: int, min_val: int = None, max_val: int = None) -> int:
    """Parse int from environment with validation.

    FIX (Error #13): Prevents crashes from invalid env var values.

    Args:
        key: Environment variable name
        default: Default value if not set or invalid
        min_val: Minimum allowed value (optional)
        max_val: Maximum allowed value (optional)

    Returns:
        Parsed and validated int value
    """
    value = os.environ.get(key)
    if value is None:
        return default

    try:
        parsed = int(value)
    except ValueError as e:
        _logger.error(
            f"Invalid integer for {key}='{value}': {e}. "
            f"Using default: {default}"
        )
        return default

    if min_val is not None and parsed < min_val:
        _logger.warning(f"{key}={parsed} below minimum {min_val}, clamping")
        return min_val
    if max_val is not None and parsed > max_val:
        _logger.warning(f"{key}={parsed} above maximum {max_val}, clamping")
        return max_val

    return parsed


def _get_bool_env(var_name: str, default: str = "1", legacy_keys: Optional[Iterable[str]] = None) -> bool:
    """Read a boolean environment variable with optional legacy aliases."""

    keys = [var_name]
    if legacy_keys:
        keys.extend(legacy_keys)

    for key in keys:
        if key in os.environ:
            value = os.environ[key]
            break
    else:
        value = default

    return value.lower() not in {"0", "false", "no", "off", ""}


# ====== OLLAMA CONFIG ======
_DEFAULT_RAG_OLLAMA_URL = "http://10.127.0.192:11434"
DEFAULT_RAG_OLLAMA_URL = _DEFAULT_RAG_OLLAMA_URL
DEFAULT_LOCAL_OLLAMA_URL = "http://127.0.0.1:11434"

RAG_OLLAMA_URL = _get_env_value(
    "RAG_OLLAMA_URL",
    default=_DEFAULT_RAG_OLLAMA_URL,
    legacy_keys=("OLLAMA_URL",),
)
RAG_CHAT_MODEL = _get_env_value(
    "RAG_CHAT_MODEL",
    default="qwen2.5:32b",
    legacy_keys=("GEN_MODEL", "CHAT_MODEL"),
)
RAG_EMBED_MODEL = _get_env_value(
    "RAG_EMBED_MODEL",
    default="nomic-embed-text:latest",
    legacy_keys=("EMB_MODEL", "EMBED_MODEL"),
)

_INITIAL_RAG_LLM_CLIENT = _get_env_value("RAG_LLM_CLIENT", default="")


@dataclass(frozen=True)
class LLMSettings:
    """Typed snapshot of the current LLM configuration."""

    base_url: str
    chat_model: str
    embed_model: str
    client_mode: str


def get_llm_client_mode(default: str = "") -> str:
    """Return the preferred LLM client mode (`mock`, `ollama`, etc.)."""

    raw_value = os.environ.get("RAG_LLM_CLIENT")
    if raw_value is None:
        raw_value = _INITIAL_RAG_LLM_CLIENT
    normalized = (raw_value or "").strip().lower()
    if not normalized:
        return default.strip().lower()
    return normalized


def current_llm_settings(default_client_mode: str = "") -> LLMSettings:
    """Return a dataclass capturing the current Ollama + model configuration."""

    return LLMSettings(
        base_url=RAG_OLLAMA_URL,
        chat_model=RAG_CHAT_MODEL,
        embed_model=RAG_EMBED_MODEL,
        client_mode=get_llm_client_mode(default_client_mode),
    )

# Backwards-compatible aliases (legacy code/tests expect these names)
OLLAMA_URL = RAG_OLLAMA_URL
GEN_MODEL = RAG_CHAT_MODEL
EMB_MODEL = RAG_EMBED_MODEL

# ====== CHUNKING CONFIG ======
CHUNK_CHARS = _parse_env_int("CHUNK_CHARS", 1600, min_val=100, max_val=8000)
CHUNK_OVERLAP = _parse_env_int("CHUNK_OVERLAP", 200, min_val=0, max_val=4000)

# ====== RETRIEVAL CONFIG ======
# OPTIMIZATION: Increase retrieval parameters for better recall on internal deployment
DEFAULT_TOP_K = _parse_env_int("DEFAULT_TOP_K", 15, min_val=1, max_val=100)  # Was 12, now 15 (more candidates)
DEFAULT_PACK_TOP = _parse_env_int("DEFAULT_PACK_TOP", 8, min_val=1, max_val=50)  # Was 6, now 8 (more snippets in context)
DEFAULT_THRESHOLD = _parse_env_float("DEFAULT_THRESHOLD", 0.25, min_val=0.0, max_val=1.0)  # Was 0.30, now 0.25 (lower bar)
DEFAULT_SEED = 42

# OPTIMIZATION: Increase max query length for internal use (no DoS risk)
MAX_QUERY_LENGTH = _parse_env_int("MAX_QUERY_LENGTH", 1000000, min_val=100, max_val=10000000)  # Was 10K, now 1M

# ====== BM25 CONFIG ======
# BM25 parameters (tuned for technical documentation)
# OPTIMIZATION: Increase k1 from 1.0 to 1.2 for slightly better term frequency saturation
# OPTIMIZATION: Keep b at 0.65 for technical docs (reduces length penalty)
# FIX (Error #13): Use safe env var parsing
BM25_K1 = _parse_env_float("BM25_K1", 1.2, min_val=0.1, max_val=10.0)  # Was 1.0, now 1.2
BM25_B = _parse_env_float("BM25_B", 0.65, min_val=0.0, max_val=1.0)

# ====== LLM CONFIG ======
# OPTIMIZATION: Increase DEFAULT_NUM_CTX to 32768 to match Qwen 32B's full context window
# This allows us to use more context for better retrieval quality
# pack_snippets enforces effective_budget = min(CTX_TOKEN_BUDGET, num_ctx * 0.6)
# With value of 32768: effective = min(12000, 19660) = 12000 ✅
# Fully utilizes Qwen 32B's 32K context window capacity
# FIX (Error #13): Use safe env var parsing
DEFAULT_NUM_CTX = _parse_env_int("DEFAULT_NUM_CTX", 32768, min_val=512, max_val=128000)  # Was 16384, now 32768
# Allow overriding generation length via env for ops tuning
DEFAULT_NUM_PREDICT = _parse_env_int("DEFAULT_NUM_PREDICT", 512, min_val=32, max_val=4096)
# FIX: Increase default retries from 0 to 2 for remote Ollama resilience
# Remote endpoints (especially over VPN) benefit from transient error retry
# Can be overridden via DEFAULT_RETRIES env var or --retries CLI flag
DEFAULT_RETRIES = _parse_env_int("DEFAULT_RETRIES", 2, min_val=0, max_val=10)  # Was 0, now 2

# ====== MMR & CONTEXT BUDGET ======
# OPTIMIZATION: Increase MMR_LAMBDA to 0.75 to favor relevance slightly over diversity
MMR_LAMBDA = _parse_env_float("MMR_LAMBDA", 0.75, min_val=0.0, max_val=1.0)  # Was 0.7, now 0.75
# OPTIMIZATION: Increase context budget from 6000 to 12000 tokens to better utilize Qwen 32B's capacity
# Qwen 32B has 32K context window; we reserve 60% for snippets (pack_snippets enforces this)
# Old: 6000 tokens (~24K chars) was still conservative
# New: 12000 tokens (~48K chars) allows 2x more context while leaving room for Q+A
# Can be overridden via CTX_BUDGET env var
# FIX (Error #13): Use safe env var parsing
CTX_TOKEN_BUDGET = _parse_env_int("CTX_BUDGET", 12000, min_val=100, max_val=100000)  # Was 6000, now 12000

# ====== EMBEDDINGS BACKEND (v4.1) ======
EMB_BACKEND = (_get_env_value("EMB_BACKEND", "local") or "local").lower()  # "local" or "ollama"

# Embedding dimensions:
# - local (SentenceTransformer all-MiniLM-L6-v2): 384-dim
# - ollama (nomic-embed-text): 768-dim
EMB_DIM_LOCAL = 384
EMB_DIM_OLLAMA = 768
EMB_DIM = EMB_DIM_LOCAL if EMB_BACKEND == "local" else EMB_DIM_OLLAMA

# ====== ANN (Approximate Nearest Neighbors) (v4.1) ======
USE_ANN = (_get_env_value("ANN", "faiss") or "faiss").lower()  # "faiss" or "none"
# Note: nlist reduced from 256→64 for arm64 macOS stability (avoid IVF training segfault)
# FIX (Error #13): Use safe env var parsing
ANN_NLIST = _parse_env_int("ANN_NLIST", 64, min_val=8, max_val=1024)  # IVF clusters (reduced for stability)
ANN_NPROBE = _parse_env_int("ANN_NPROBE", 16, min_val=1, max_val=256)  # clusters to search

# ====== HYBRID SCORING (v4.1) ======
# FIX (Error #13): Use safe env var parsing
ALPHA_HYBRID = _parse_env_float("ALPHA", 0.5, min_val=0.0, max_val=1.0)  # 0.5 = BM25 and dense equally weighted (fallback)

# ====== INTENT CLASSIFICATION (v5.9) ======
# OPTIMIZATION: Enable intent-based retrieval for +8-12% accuracy improvement
# When enabled, alpha_hybrid is dynamically adjusted based on query intent:
# - Procedural (how-to): 0.65 (favor BM25 for keyword matching)
# - Factual (what/define): 0.35 (favor dense for semantic understanding)
# - Pricing: 0.70 (high BM25 for exact pricing terms)
# - Troubleshooting: 0.60 (favor BM25 for error messages)
# - General: 0.50 (balanced, same as ALPHA_HYBRID)
USE_INTENT_CLASSIFICATION = _get_bool_env("USE_INTENT_CLASSIFICATION", "1")

# ====== KPI TIMINGS (v4.1) ======
class KPI:
    """Global KPI tracking for performance metrics."""
    retrieve_ms = 0
    ann_ms = 0
    rerank_ms = 0
    ask_ms = 0


# ====== TIMEOUT CONFIG ======
# Task G: Deterministic timeouts (environment-configurable for ops)
# FIX (Error #13): Use safe env var parsing
EMB_CONNECT_T = _parse_env_float("EMB_CONNECT_TIMEOUT", 3.0, min_val=0.1, max_val=60.0)
EMB_READ_T = _parse_env_float("EMB_READ_TIMEOUT", 60.0, min_val=1.0, max_val=600.0)
CHAT_CONNECT_T = _parse_env_float("CHAT_CONNECT_TIMEOUT", 3.0, min_val=0.1, max_val=60.0)
CHAT_READ_T = _parse_env_float("CHAT_READ_TIMEOUT", 120.0, min_val=1.0, max_val=600.0)
RERANK_READ_T = _parse_env_float("RERANK_READ_TIMEOUT", 180.0, min_val=1.0, max_val=600.0)

# ====== EMBEDDING BATCHING CONFIG (Rank 10) ======
# Parallel embedding generation for faster KB builds (3-5x speedup)
# FIX (Error #13): Use safe env var parsing
EMB_MAX_WORKERS = _parse_env_int("EMB_MAX_WORKERS", 8, min_val=1, max_val=64)  # Concurrent requests
EMB_BATCH_SIZE = _parse_env_int("EMB_BATCH_SIZE", 32, min_val=1, max_val=1000)  # Texts per batch

# ====== REFUSAL STRING ======
# Exact refusal string (ASCII quotes only)
REFUSAL_STR = "I don't know based on the MD."

# ====== LOGGING CONFIG ======


# Query logging configuration
QUERY_LOG_FILE = _get_env_value("RAG_LOG_FILE", "rag_queries.jsonl") or "rag_queries.jsonl"
LOG_QUERY_INCLUDE_ANSWER = _get_bool_env("RAG_LOG_INCLUDE_ANSWER", "1")
LOG_QUERY_ANSWER_PLACEHOLDER = _get_env_value("RAG_LOG_ANSWER_PLACEHOLDER", "[REDACTED]") or "[REDACTED]"
LOG_QUERY_INCLUDE_CHUNKS = _get_bool_env("RAG_LOG_INCLUDE_CHUNKS", "0")  # Redact chunk text by default for security/privacy

# Citation validation configuration
STRICT_CITATIONS = _get_bool_env("RAG_STRICT_CITATIONS", "0")  # Refuse answers without citations (improves trust in regulated environments)

# ====== CACHING & RATE LIMITING CONFIG ======
# Query cache size
CACHE_MAXSIZE = _parse_env_int("CACHE_MAXSIZE", 100, min_val=1, max_val=10000)
# Cache TTL in seconds
CACHE_TTL = _parse_env_int("CACHE_TTL", 3600, min_val=60, max_val=86400)
# Rate limiting: max requests per window
RATE_LIMIT_REQUESTS = _parse_env_int("RATE_LIMIT_REQUESTS", 10, min_val=1, max_val=1000)
# Rate limiting window in seconds
RATE_LIMIT_WINDOW = _parse_env_int("RATE_LIMIT_WINDOW", 60, min_val=1, max_val=3600)

# ====== API AUTH CONFIG ======
API_AUTH_MODE = (_get_env_value("API_AUTH_MODE", "none") or "none").strip().lower()
_api_keys_raw = _get_env_value("API_ALLOWED_KEYS", "")
if _api_keys_raw.strip():
    API_ALLOWED_KEYS = frozenset(
        key.strip() for key in _api_keys_raw.split(",") if key.strip()
    )
else:
    API_ALLOWED_KEYS = frozenset()
API_KEY_HEADER = (_get_env_value("API_KEY_HEADER", "x-api-key") or "x-api-key").strip() or "x-api-key"

# ====== WARMUP CONFIG ======
# Warm-up on startup
WARMUP_ENABLED = _get_bool_env("WARMUP", "1")

# ====== NLTK DOWNLOAD CONFIG ======
# Auto-download NLTK data
NLTK_AUTO_DOWNLOAD = _get_bool_env("NLTK_AUTO_DOWNLOAD", "1")

# ====== QUERY EXPANSION CONFIG ======
# Query expansion file path
CLOCKIFY_QUERY_EXPANSIONS = _get_env_value("CLOCKIFY_QUERY_EXPANSIONS", None)

# Maximum query expansion file size (in bytes)
MAX_QUERY_EXPANSION_FILE_SIZE = _parse_env_int("MAX_QUERY_EXPANSION_FILE_SIZE", 10485760, min_val=1024, max_val=104857600)  # 10MB default, 100MB max

# ====== PROXY CONFIGURATION ======
# Optional HTTP proxy support (disabled by default for security)
ALLOW_PROXIES = _get_bool_env("ALLOW_PROXIES", "0", legacy_keys=("USE_PROXY",))  # Enable proxy usage when set to 1/true/yes
HTTP_PROXY = _get_env_value("HTTP_PROXY", "") or ""
HTTPS_PROXY = _get_env_value("HTTPS_PROXY", "") or ""

# Set proxy environment variables if allowed and configured
if ALLOW_PROXIES:
    if HTTP_PROXY:
        os.environ["HTTP_PROXY"] = HTTP_PROXY
    if HTTPS_PROXY:
        os.environ["HTTPS_PROXY"] = HTTPS_PROXY

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
# FIX (Error #13): Use safe env var parsing
BUILD_LOCK_TTL_SEC = _parse_env_int("BUILD_LOCK_TTL_SEC", 900, min_val=60, max_val=7200)  # Task D: 15 minutes default

# ====== RETRIEVAL CONFIG (CONTINUED) ======
# FAISS/HNSW candidate generation (Quick Win #6)
# Expose FAISS candidate knobs through env for prod-level tuning
FAISS_CANDIDATE_MULTIPLIER = _parse_env_int("FAISS_CANDIDATE_MULTIPLIER", 3, min_val=1, max_val=10)  # Retrieve top_k * N
ANN_CANDIDATE_MIN = _parse_env_int("ANN_CANDIDATE_MIN", 200, min_val=1, max_val=2000)  # Minimum candidates even if top_k is small

# Reranking (Quick Win #6)
RERANK_SNIPPET_MAX_CHARS = 500  # Truncate chunk text for reranking prompt
RERANK_MAX_CHUNKS = 12  # Maximum chunks to send to reranking

# Retrieval thresholds (Quick Win #6)
COVERAGE_MIN_CHUNKS = 2  # Minimum chunks above threshold to proceed

# ====== PRECOMPUTED FAQ CACHE (Analysis Section 9.1 #3) ======
# OPTIMIZATION: Pre-generate answers for top FAQs for 100% cache hit on common queries
FAQ_CACHE_ENABLED = _get_bool_env("FAQ_CACHE_ENABLED", "0")  # Disabled by default (requires build step)
FAQ_CACHE_PATH = _get_env_value("FAQ_CACHE_PATH", "faq_cache.json") or "faq_cache.json"
