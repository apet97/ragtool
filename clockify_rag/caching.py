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
        self.cache: dict = {}  # {question_hash: (answer, metadata, timestamp)}
        self.access_order: deque = deque()  # For LRU eviction
        self.hits = 0
        self.misses = 0
        self._lock = threading.RLock()  # Thread safety lock

    def _hash_question(self, question: str) -> str:
        """Generate cache key from question."""
        return hashlib.md5(question.encode('utf-8')).hexdigest()

    def get(self, question: str):
        """Retrieve cached answer if available and not expired.

        Returns:
            (answer, metadata) tuple if cache hit, None if cache miss
        """
        with self._lock:
            key = self._hash_question(question)

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

    def put(self, question: str, answer: str, metadata: dict):
        """Store answer in cache.

        Args:
            question: User question
            answer: Generated answer
            metadata: Answer metadata (selected chunks, scores, etc.)
        """
        with self._lock:
            key = self._hash_question(question)

            # Evict oldest entry if cache full
            if len(self.cache) >= self.maxsize and key not in self.cache:
                oldest = self.access_order.popleft()
                del self.cache[oldest]
                logger.debug(f"[cache] EVICT question_hash={oldest[:8]} (LRU)")

            # Store entry with timestamp
            self.cache[key] = (answer, metadata, time.time())

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
    from .config import QUERY_LOG_FILE

    # Extract chunk IDs and scores
    chunk_ids = [c["id"] if isinstance(c, dict) else c for c in retrieved_chunks]
    chunk_scores = [c.get("score", 0.0) if isinstance(c, dict) else 0.0 for c in retrieved_chunks]

    log_entry = {
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query": query,
        "answer": answer,
        "refused": refused,
        "latency_ms": latency_ms,
        "retrieved_chunks": len(chunk_ids),
        "chunk_ids": chunk_ids,
        "chunk_scores": chunk_scores,
        "metadata": metadata or {}
    }

    try:
        with open(QUERY_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to log query: {e}")
