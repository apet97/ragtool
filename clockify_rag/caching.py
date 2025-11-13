"""Query caching and rate limiting for RAG system."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional, Callable

from logging.handlers import RotatingFileHandler
from pathlib import Path

try:  # Optional dependency for distributed rate limiting
    import redis
    from redis import Redis, RedisError
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    redis = None
    Redis = None
    RedisError = Exception

logger = logging.getLogger(__name__)

# FIX (Error #2): Declare globals at module level for safe initialization
_RATE_LIMITER = None
_QUERY_CACHE = None
_QUERY_LOGGER = None
_QUERY_LOGGER_LOCK = threading.Lock()


class RateLimitError(RuntimeError):
    """Raised when the rate limiter backend cannot be used."""


@dataclass(frozen=True)
class RateLimitSettings:
    """Configuration for a single sliding-window limit."""

    max_requests: int
    window_seconds: float

    @property
    def disabled(self) -> bool:
        return self.max_requests <= 0 or self.window_seconds <= 0


class RateLimiter:
    """Sliding-window rate limiter keyed by identity."""

    _GLOBAL_KEY = "__global__"

    def __init__(
        self,
        max_requests: int = 10,
        window_seconds: float = 60.0,
        global_limit: Optional[RateLimitSettings] = None,
    ):
        if max_requests < 0:
            raise ValueError("max_requests must be >= 0")
        if window_seconds < 0:
            raise ValueError("window_seconds must be >= 0")

        self.max_requests = int(max_requests)
        self.window_seconds = float(window_seconds)
        self._identity_disabled = self.max_requests == 0 or self.window_seconds == 0
        if global_limit and global_limit.disabled:
            global_limit = None
        self._global_limit = global_limit
        self._time_fn = time.monotonic
        self._lock = threading.RLock()
        self._events = defaultdict(deque)  # identity -> deque[timestamps]
        self._global_events = deque()

    def _normalized_key(self, identity: Optional[str]) -> str:
        return identity or self._GLOBAL_KEY

    def _prune_identity(self, key: str, now: float) -> deque:
        bucket = self._events.get(key)
        if not bucket:
            return deque()
        cutoff = now - self.window_seconds
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if not bucket:
            self._events.pop(key, None)
            return deque()
        return bucket

    def _prune_global(self, now: float) -> deque:
        if not self._global_limit:
            return deque()
        cutoff = now - self._global_limit.window_seconds
        bucket = self._global_events
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        return bucket

    def allow_request(self, identity: Optional[str] = None) -> bool:
        """Return True if request is allowed for the identity."""

        if self._identity_disabled and not self._global_limit:
            return True

        now = self._time_fn()
        key = self._normalized_key(identity)

        with self._lock:
            bucket = None
            if not self._identity_disabled:
                bucket = self._prune_identity(key, now)
                if len(bucket) >= self.max_requests:
                    return False

            global_bucket = None
            if self._global_limit:
                global_bucket = self._prune_global(now)
                if len(global_bucket) >= self._global_limit.max_requests:
                    return False

            if not self._identity_disabled:
                bucket = bucket or self._prune_identity(key, now)
                bucket.append(now)
                self._events[key] = bucket
            if self._global_limit:
                global_bucket = global_bucket or self._prune_global(now)
                global_bucket.append(now)

            return True

    def wait_time(self, identity: Optional[str] = None) -> float:
        """Return seconds until the next request would be allowed."""

        if self._identity_disabled and not self._global_limit:
            return 0.0

        now = self._time_fn()
        key = self._normalized_key(identity)

        with self._lock:
            wait_candidates = []

            if not self._identity_disabled:
                bucket = self._prune_identity(key, now)
                if len(bucket) >= self.max_requests:
                    oldest = bucket[0]
                    retry_after = (oldest + self.window_seconds) - now
                    wait_candidates.append(retry_after)

            if self._global_limit:
                global_bucket = self._prune_global(now)
                if len(global_bucket) >= self._global_limit.max_requests:
                    oldest = global_bucket[0]
                    retry_after = (oldest + self._global_limit.window_seconds) - now
                    wait_candidates.append(retry_after)

            if not wait_candidates:
                return 0.0

            return max(0.0, max(wait_candidates))


class RedisRateLimiter:
    """Distributed limiter backed by Redis sorted sets."""

    def __init__(
        self,
        client: Redis,
        identity_limit: RateLimitSettings,
        global_limit: Optional[RateLimitSettings] = None,
        namespace: str = "clockify:rate",
        max_retries: int = 2,
    ) -> None:
        if client is None:
            raise ValueError("Redis client is required for RedisRateLimiter")
        self.client = client
        self.identity_limit = identity_limit
        self.global_limit = None if (global_limit and global_limit.disabled) else global_limit
        self.namespace = namespace or "clockify:rate"
        self.max_retries = max(1, int(max_retries or 1))

    def _identity_disabled(self) -> bool:
        return self.identity_limit.disabled if self.identity_limit else True

    def _key(self, identity: Optional[str], settings: RateLimitSettings) -> str:
        normalized = identity or "anonymous"
        safe = normalized.strip() or "anonymous"
        if len(safe) > 64:
            safe = hashlib.sha1(safe.encode("utf-8")).hexdigest()
        return f"{self.namespace}:{int(settings.window_seconds)}:{safe}"

    def _run_with_retries(self, func: Callable[[], float | bool | list]) -> float | bool | list:
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                return func()
            except RedisError as exc:  # pragma: no cover - network failure is rare
                last_exc = exc
                backoff = min(0.25, 0.05 * (attempt + 1))
                time.sleep(backoff)
        raise RateLimitError(f"Redis backend unavailable: {last_exc}") from last_exc

    def _within_limit(self, key: str, settings: RateLimitSettings, now: float) -> bool:
        if settings.disabled:
            return True

        cutoff = now - settings.window_seconds

        def op():
            pipe = self.client.pipeline()
            pipe.zremrangebyscore(key, 0, cutoff)
            pipe.zcard(key)
            return pipe.execute()

        _, count = self._run_with_retries(op)
        return count < settings.max_requests

    def _record_event(self, key: str, settings: RateLimitSettings, now: float) -> None:
        if settings.disabled:
            return

        def op():
            pipe = self.client.pipeline()
            pipe.zadd(key, {str(now): now})
            pipe.expire(key, int(settings.window_seconds) + 1)
            return pipe.execute()

        self._run_with_retries(op)

    def allow_request(self, identity: Optional[str] = None) -> bool:
        if self._identity_disabled() and not self.global_limit:
            return True

        now = time.time()
        identity_allowed = True
        if not self._identity_disabled():
            identity_key = self._key(identity, self.identity_limit)
            identity_allowed = self._within_limit(identity_key, self.identity_limit, now)

        global_allowed = True
        if self.global_limit:
            global_key = self._key("__global__", self.global_limit)
            global_allowed = self._within_limit(global_key, self.global_limit, now)

        if identity_allowed and global_allowed:
            if not self._identity_disabled():
                identity_key = self._key(identity, self.identity_limit)
                self._record_event(identity_key, self.identity_limit, now)
            if self.global_limit:
                global_key = self._key("__global__", self.global_limit)
                self._record_event(global_key, self.global_limit, now)
            return True
        return False

    def _wait_time_for_key(self, key: str, settings: RateLimitSettings, now: float) -> float:
        if settings.disabled:
            return 0.0
        cutoff = now - settings.window_seconds

        def op():
            pipe = self.client.pipeline()
            pipe.zremrangebyscore(key, 0, cutoff)
            pipe.zrange(key, 0, 0, withscores=True)
            return pipe.execute()

        _, entries = self._run_with_retries(op)
        if not entries:
            return 0.0
        oldest_score = entries[0][1]
        retry_after = (oldest_score + settings.window_seconds) - now
        return max(0.0, retry_after)

    def wait_time(self, identity: Optional[str] = None) -> float:
        if self._identity_disabled() and not self.global_limit:
            return 0.0

        now = time.time()
        waits = []
        if not self._identity_disabled():
            identity_key = self._key(identity, self.identity_limit)
            waits.append(self._wait_time_for_key(identity_key, self.identity_limit, now))
        if self.global_limit:
            global_key = self._key("__global__", self.global_limit)
            waits.append(self._wait_time_for_key(global_key, self.global_limit, now))

        return max(waits) if waits else 0.0


# Global rate limiter (10 queries per minute by default)
_RATE_LIMITER_LOCK = threading.Lock()


def _parse_int(env_keys: list[str], default: int) -> int:
    for key in env_keys:
        raw = os.environ.get(key)
        if raw is None:
            continue
        try:
            return int(raw)
        except ValueError:
            logger.warning("Invalid integer for %s='%s'. Using %s.", key, raw, default)
            break
    return default


def _parse_float(env_keys: list[str], default: float) -> float:
    for key in env_keys:
        raw = os.environ.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except ValueError:
            logger.warning("Invalid float for %s='%s'. Using %s.", key, raw, default)
            break
    return default


def _build_limit_settings(request_env: list[str], window_env: list[str], default_requests: int, default_window: float) -> RateLimitSettings:
    requests = _parse_int(request_env, default_requests)
    window = _parse_float(window_env, default_window)
    return RateLimitSettings(max_requests=requests, window_seconds=window)


def _create_rate_limiter_from_env():
    identity_limit = _build_limit_settings(
        ["RATE_LIMIT_IDENTITY_REQUESTS", "RATE_LIMIT_REQUESTS"],
        ["RATE_LIMIT_IDENTITY_WINDOW", "RATE_LIMIT_WINDOW"],
        default_requests=10,
        default_window=60.0,
    )

    global_limit_raw = _build_limit_settings(
        ["RATE_LIMIT_GLOBAL_REQUESTS"],
        ["RATE_LIMIT_GLOBAL_WINDOW"],
        default_requests=0,
        default_window=60.0,
    )
    global_limit = None if global_limit_raw.disabled else global_limit_raw

    backend = (os.environ.get("RATE_LIMIT_BACKEND", "memory") or "memory").strip().lower()
    namespace = os.environ.get("RATE_LIMIT_NAMESPACE", "clockify:rate")

    if backend == "redis":
        if redis is None:
            logger.warning("RATE_LIMIT_BACKEND=redis but redis package not installed. Falling back to in-memory limiter.")
        else:
            redis_url = os.environ.get("RATE_LIMIT_REDIS_URL", "redis://127.0.0.1:6379/0")
            socket_timeout = _parse_float(["RATE_LIMIT_REDIS_TIMEOUT"], 1.0)
            max_retries = _parse_int(["RATE_LIMIT_REDIS_RETRIES"], 2)
            try:
                client = redis.Redis.from_url(
                    redis_url,
                    socket_timeout=socket_timeout,
                    socket_connect_timeout=socket_timeout,
                    retry_on_timeout=True,
                )
                client.ping()
                logger.info("Using Redis-backed rate limiter (backend=%s, namespace=%s)", redis_url, namespace)
                return RedisRateLimiter(
                    client,
                    identity_limit,
                    global_limit=global_limit,
                    namespace=namespace,
                    max_retries=max_retries,
                )
            except RedisError as exc:
                logger.error("Failed to initialize Redis rate limiter at %s: %s. Falling back to in-memory limiter.", redis_url, exc)

    if not identity_limit.disabled:
        logger.info("Using in-memory rate limiter (window=%ss, max=%s)", identity_limit.window_seconds, identity_limit.max_requests)
    elif global_limit:
        logger.info("Using in-memory global-only rate limiter (window=%ss, max=%s)", global_limit.window_seconds, global_limit.max_requests)
    return RateLimiter(
        max_requests=identity_limit.max_requests,
        window_seconds=identity_limit.window_seconds,
        global_limit=global_limit,
    )


def get_rate_limiter():
    """Get global rate limiter instance."""

    global _RATE_LIMITER
    if _RATE_LIMITER is None:
        with _RATE_LIMITER_LOCK:
            if _RATE_LIMITER is None:
                _RATE_LIMITER = _create_rate_limiter_from_env()
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
        # FIX (Error #4): Add maxlen as defense-in-depth safety net
        # maxlen = maxsize * 2 provides safety buffer if cleanup fails
        self.access_order: deque = deque(maxlen=maxsize * 2)  # For LRU eviction
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
            # FIX: Deep copy metadata to prevent mutation leaks
            import copy
            timestamp = time.time()
            metadata_copy = copy.deepcopy(metadata) if metadata is not None else {}
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

    def save(self, path: str = "query_cache.json"):
        """Save cache to disk for persistence across sessions.

        OPTIMIZATION: Enables 100% cache hit rate on repeated queries after restart.

        Args:
            path: File path to save cache (default: query_cache.json)
        """
        import json
        with self._lock:
            try:
                cache_data = {
                    "version": "1.0",
                    "maxsize": self.maxsize,
                    "ttl_seconds": self.ttl_seconds,
                    "entries": [
                        {
                            "key": key,
                            "answer": answer,
                            "metadata": metadata,
                            "timestamp": timestamp
                        }
                        for key, (answer, metadata, timestamp) in self.cache.items()
                    ],
                    "access_order": list(self.access_order),
                    "hits": self.hits,
                    "misses": self.misses
                }
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
                logger.info(f"[cache] SAVE {len(self.cache)} entries to {path}")
            except Exception as e:
                logger.warning(f"[cache] Failed to save cache: {e}")

    def load(self, path: str = "query_cache.json"):
        """Load cache from disk to restore across sessions.

        OPTIMIZATION: Restores previous session's cache for instant hits on repeated queries.

        Args:
            path: File path to load cache from (default: query_cache.json)

        Returns:
            Number of entries loaded (0 if file doesn't exist or load fails)
        """
        import json
        with self._lock:
            if not os.path.exists(path):
                logger.debug(f"[cache] No cache file found at {path}")
                return 0

            try:
                with open(path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)

                # Validate version
                version = cache_data.get("version", "1.0")
                if version != "1.0":
                    logger.warning(f"[cache] Incompatible cache version {version}, skipping load")
                    return 0

                # Restore entries, filtering out expired ones
                now = time.time()
                loaded_count = 0
                for entry in cache_data.get("entries", []):
                    key = entry["key"]
                    answer = entry["answer"]
                    metadata = entry["metadata"]
                    timestamp = entry["timestamp"]

                    # Skip expired entries
                    age = now - timestamp
                    if age > self.ttl_seconds:
                        continue

                    self.cache[key] = (answer, metadata, timestamp)
                    loaded_count += 1

                # Restore access order (only for non-expired keys)
                self.access_order = deque(
                    [k for k in cache_data.get("access_order", []) if k in self.cache],
                    maxlen=self.maxsize * 2
                )

                # Restore stats (reset to avoid inflated numbers from old sessions)
                # self.hits = cache_data.get("hits", 0)
                # self.misses = cache_data.get("misses", 0)

                logger.info(f"[cache] LOAD {loaded_count} entries from {path} (skipped {len(cache_data.get('entries', [])) - loaded_count} expired)")
                return loaded_count

            except Exception as e:
                logger.warning(f"[cache] Failed to load cache: {e}")
                return 0


# Global query cache (100 entries, 1 hour TTL by default)
def get_query_cache():
    """Get global query cache instance.

    FIX (Error #2): Use proper `is None` check instead of fragile globals() check.
    """
    global _QUERY_CACHE
    if _QUERY_CACHE is None:
        _QUERY_CACHE = QueryCache(
            maxsize=int(os.environ.get("CACHE_MAXSIZE", "100")),
            ttl_seconds=int(os.environ.get("CACHE_TTL", "3600"))
        )
    return _QUERY_CACHE


def _get_query_logger() -> logging.Logger:
    """Return a dedicated logger for structured query logs."""

    global _QUERY_LOGGER
    if _QUERY_LOGGER is not None:
        return _QUERY_LOGGER

    with _QUERY_LOGGER_LOCK:
        if _QUERY_LOGGER is not None:
            return _QUERY_LOGGER

        from .config import (
            QUERY_LOG_BACKUP_COUNT,
            QUERY_LOG_FILE,
            QUERY_LOG_MAX_BYTES,
        )

        log_path = Path(QUERY_LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        handler = RotatingFileHandler(
            log_path,
            maxBytes=int(QUERY_LOG_MAX_BYTES),
            backupCount=int(QUERY_LOG_BACKUP_COUNT),
            encoding="utf-8",
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        setattr(handler, "_clockify_query_handler", True)

        query_logger = logging.getLogger("clockify_rag.query_log")
        query_logger.setLevel(logging.INFO)
        query_logger.propagate = False

        for existing in list(query_logger.handlers):
            if getattr(existing, "_clockify_query_handler", False):
                query_logger.removeHandler(existing)
                try:
                    existing.close()
                except Exception:  # pragma: no cover - best effort cleanup
                    pass

        query_logger.addHandler(handler)

        _QUERY_LOGGER = query_logger
        return _QUERY_LOGGER


def _reset_query_logger_for_tests() -> None:
    """Reset the cached query logger so tests can reconfigure it."""

    global _QUERY_LOGGER
    with _QUERY_LOGGER_LOCK:
        if _QUERY_LOGGER is None:
            return

        for handler in list(_QUERY_LOGGER.handlers):
            if getattr(handler, "_clockify_query_handler", False):
                _QUERY_LOGGER.removeHandler(handler)
                try:
                    handler.close()
                except Exception:  # pragma: no cover - best effort cleanup
                    pass

        _QUERY_LOGGER = None


def log_query(query: str, answer: str, retrieved_chunks: list, latency_ms: float,
              refused: bool = False, metadata: dict = None):
    """Log query with structured JSON format for monitoring and analytics.

    FIX (Error #6): Sanitizes user input to prevent log injection attacks.
    """
    import json
    from .config import (
        LOG_QUERY_ANSWER_PLACEHOLDER,
        LOG_QUERY_INCLUDE_ANSWER,
        LOG_QUERY_INCLUDE_CHUNKS,
        QUERY_LOG_FILE,
    )
    from .utils import sanitize_for_log

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

    # FIX: Sanitize metadata to prevent chunk text leaks
    # Deep copy and remove any 'text'/'chunk' fields from nested structures
    import copy
    sanitized_metadata = copy.deepcopy(metadata) if metadata else {}
    if not LOG_QUERY_INCLUDE_CHUNKS and isinstance(sanitized_metadata, dict):
        # Remove chunk text from any nested chunk dicts in metadata
        for key in list(sanitized_metadata.keys()):
            val = sanitized_metadata[key]
            if isinstance(val, dict):
                val.pop("text", None)
                val.pop("chunk", None)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        item.pop("text", None)
                        item.pop("chunk", None)

    # FIX (Error #6): Sanitize query and answer to prevent log injection
    log_entry = {
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query": sanitize_for_log(query, max_length=2000),
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
        "metadata": sanitized_metadata,
    }

    if LOG_QUERY_INCLUDE_ANSWER:
        log_entry["answer"] = sanitize_for_log(answer, max_length=5000)
    elif LOG_QUERY_ANSWER_PLACEHOLDER:
        log_entry["answer"] = LOG_QUERY_ANSWER_PLACEHOLDER

    try:
        query_logger = _get_query_logger()
    except Exception as exc:  # pragma: no cover - initialization failures are rare
        logger.warning("Failed to initialize query logger: %s", exc)
        return

    try:
        query_logger.info(json.dumps(log_entry, ensure_ascii=False))
    except Exception as e:  # pragma: no cover - handler failure is unexpected
        logger.warning(f"Failed to log query: {e}")

