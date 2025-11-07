"""Query caching and rate limiting for RAG system."""

import hashlib
import logging
import os
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)


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
        self.requests: deque = deque()
        self._lock = threading.RLock()  # Thread safety lock

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
def get_rate_limiter():
    """Get global rate limiter instance."""
    global _RATE_LIMITER
    if '_RATE_LIMITER' not in globals():
        _RATE_LIMITER = RateLimiter(
            max_requests=int(os.environ.get("RATE_LIMIT_REQUESTS", "10")),
            window_seconds=int(os.environ.get("RATE_LIMIT_WINDOW", "60"))
        )
    return _RATE_LIMITER


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
        self.cache: dict = {}  # {question_hash: (answer, metadata_with_timestamp, timestamp)}
        self.access_order: deque = deque()  # For LRU eviction
        self.hits = 0
        self.misses = 0
        self._lock = threading.RLock()  # Thread safety lock

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
            # Ensure metadata exposes cache timestamp for downstream logging
            metadata_timestamp = metadata.get("timestamp")
            if metadata_timestamp is None:
                metadata_timestamp = timestamp
                metadata["timestamp"] = metadata_timestamp

            age = time.time() - metadata_timestamp

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

            # Store entry with timestamp
            timestamp = time.time()
            metadata = dict(metadata) if metadata is not None else {}
            metadata["timestamp"] = timestamp
            self.cache[key] = (answer, metadata, timestamp)

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
def get_query_cache():
    """Get global query cache instance."""
    global _QUERY_CACHE
    if '_QUERY_CACHE' not in globals():
        _QUERY_CACHE = QueryCache(
            maxsize=int(os.environ.get("CACHE_MAXSIZE", "100")),
            ttl_seconds=int(os.environ.get("CACHE_TTL", "3600"))
        )
    return _QUERY_CACHE


def log_query(query: str, answer: str, retrieved_chunks: list, latency_ms: float,
              refused: bool = False, metadata: dict = None):
    """Log query with structured JSON format for monitoring and analytics."""
    import json
    from .config import (
        LOG_QUERY_ANSWER_PLACEHOLDER,
        LOG_QUERY_INCLUDE_ANSWER,
        LOG_QUERY_INCLUDE_CHUNKS,
        QUERY_LOG_FILE,
    )

    normalized_chunks = []
    for chunk in retrieved_chunks:
        if isinstance(chunk, dict):
            normalized = chunk.copy()
            chunk_id = normalized.get("id") or normalized.get("chunk_id")
            normalized["id"] = chunk_id
            normalized["dense"] = float(normalized.get("dense", normalized.get("score", 0.0)))
            normalized["bm25"] = float(normalized.get("bm25", 0.0))
            normalized["hybrid"] = float(normalized.get("hybrid", normalized["dense"]))
            # Redact chunk text for security/privacy unless explicitly enabled
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
    avg_chunk_score = (sum(primary_scores) / len(primary_scores)) if primary_scores else 0.0
    max_chunk_score = max(primary_scores) if primary_scores else 0.0

    log_entry = {
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query": query,
        "refused": refused,
        "latency_ms": latency_ms,
        "num_chunks_retrieved": len(chunk_ids),
        "chunk_ids": chunk_ids,
        "chunk_scores": {
            "dense": dense_scores,
            "bm25": bm25_scores,
            "hybrid": hybrid_scores,
        },
        "retrieved_chunks": normalized_chunks,
        "avg_chunk_score": avg_chunk_score,
        "max_chunk_score": max_chunk_score,
        "metadata": metadata or {},
    }

    if LOG_QUERY_INCLUDE_ANSWER:
        log_entry["answer"] = answer
    elif LOG_QUERY_ANSWER_PLACEHOLDER:
        log_entry["answer"] = LOG_QUERY_ANSWER_PLACEHOLDER

    try:
        with open(QUERY_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to log query: {e}")

