"""Clockify RAG System - Modular Package

A production-ready Retrieval-Augmented Generation system with:
- Hybrid retrieval (BM25 + dense + MMR)
- Plugin architecture for extensibility
- Comprehensive caching and rate limiting
- Full offline operation
"""

__version__ = "5.0.0"

# Core exceptions
from .exceptions import (
    EmbeddingError,
    LLMError,
    IndexLoadError,
    BuildError
)

# Configuration
from .config import (
    OLLAMA_URL,
    GEN_MODEL,
    EMB_MODEL,
    CHUNK_CHARS,
    DEFAULT_TOP_K,
    DEFAULT_PACK_TOP,
    REFUSAL_STR
)

# Utility functions
from .utils import (
    validate_ollama_url,
    validate_and_set_config,
    log_event,
    compute_sha256
)

# Chunking
from .chunking import (
    parse_articles,
    build_chunks,
    sliding_chunks
)

# Embedding
from .embedding import (
    embed_texts,
    embed_local_batch,
    validate_ollama_embeddings
)

# Indexing
from .indexing import (
    build,
    load_index,
    build_bm25,
    bm25_scores,
    build_faiss_index
)

# Caching
from .caching import (
    QueryCache,
    RateLimiter,
    get_query_cache,
    get_rate_limiter
)

__all__ = [
    # Version
    "__version__",
    # Exceptions
    "EmbeddingError", "LLMError", "IndexLoadError", "BuildError",
    # Config
    "OLLAMA_URL", "GEN_MODEL", "EMB_MODEL", "CHUNK_CHARS",
    "DEFAULT_TOP_K", "DEFAULT_PACK_TOP", "REFUSAL_STR",
    # Utils
    "validate_ollama_url", "validate_and_set_config",
    "log_event", "compute_sha256",
    # Chunking
    "parse_articles", "build_chunks", "sliding_chunks",
    # Embedding
    "embed_texts", "embed_local_batch", "validate_ollama_embeddings",
    # Indexing
    "build", "load_index", "build_bm25", "bm25_scores", "build_faiss_index",
    # Caching
    "QueryCache", "RateLimiter", "get_query_cache", "get_rate_limiter",
]
