"""Embedding generation using local SentenceTransformer or Ollama API."""

import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

from .config import EMB_MODEL, EMB_BACKEND, EMB_DIM, OLLAMA_URL, EMB_CONNECT_T, EMB_READ_T, FILES, EMB_MAX_WORKERS, EMB_BATCH_SIZE
from .exceptions import EmbeddingError
from .http_utils import get_session

logger = logging.getLogger(__name__)

# Global state for lazy-loaded sentence transformer
_ST_ENCODER = None
_ST_BATCH_SIZE = 32


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


def _embed_single_text(index: int, text: str, retries: int, total: int) -> tuple:
    """Embed a single text using Ollama API (helper for parallel batching).

    Args:
        index: Index of this text in the full list
        text: Text to embed
        retries: Number of retries for HTTP session
        total: Total number of texts (for logging)

    Returns:
        tuple: (index, embedding_list) or raises EmbeddingError
    """
    # Rank 5 fix: Create thread-local session to avoid sharing across threads
    sess = get_session(retries=retries)
    try:
        r = sess.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMB_MODEL, "prompt": text},
            timeout=(EMB_CONNECT_T, EMB_READ_T),
            allow_redirects=False
        )
        r.raise_for_status()

        # Validate embedding is not empty
        resp_json = r.json()
        emb = resp_json.get("embedding", [])
        if not emb or len(emb) == 0:
            raise EmbeddingError(f"Embedding chunk {index}: empty embedding returned (check Ollama API format)")

        return (index, emb)
    except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
        raise EmbeddingError(f"Embedding chunk {index} failed: {e} [hint: check OLLAMA_URL or increase EMB timeouts]") from e
    except requests.exceptions.RequestException as e:
        raise EmbeddingError(f"Embedding chunk {index} request failed: {e}") from e
    except EmbeddingError:
        raise  # Re-raise EmbeddingError
    except Exception as e:
        raise EmbeddingError(f"Embedding chunk {index}: {e}") from e


def embed_texts(texts: list, retries=0) -> np.ndarray:
    """Embed texts using Ollama with parallel batching (Rank 10: 3-5x speedup).

    Uses ThreadPoolExecutor to send multiple embedding requests concurrently.
    Falls back to sequential processing for small batches or on error.
    """
    if len(texts) == 0:
        return np.zeros((0, EMB_DIM), dtype="float32")

    total = len(texts)

    # Rank 10: Use parallel batching for speedup (3-5x faster KB builds)
    # Only use batching if we have enough texts to benefit from parallelism
    use_batching = total >= EMB_BATCH_SIZE and EMB_MAX_WORKERS > 1

    if not use_batching:
        # Sequential fallback for small batches or single-threaded mode
        sess = get_session(retries=retries)
        vecs = []
        for i, t in enumerate(texts):
            if (i + 1) % 100 == 0:
                logger.info(f"  [{i + 1}/{total}]")
            try:
                r = sess.post(
                    f"{OLLAMA_URL}/api/embeddings",
                    json={"model": EMB_MODEL, "prompt": t},
                    timeout=(EMB_CONNECT_T, EMB_READ_T),
                    allow_redirects=False
                )
                r.raise_for_status()

                resp_json = r.json()
                emb = resp_json.get("embedding", [])
                if not emb or len(emb) == 0:
                    raise EmbeddingError(f"Embedding chunk {i}: empty embedding returned")
                vecs.append(emb)
            except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                raise EmbeddingError(f"Embedding chunk {i} failed: {e} [hint: check OLLAMA_URL or increase EMB timeouts]") from e
            except requests.exceptions.RequestException as e:
                raise EmbeddingError(f"Embedding chunk {i} request failed: {e}") from e
            except EmbeddingError:
                raise
            except Exception as e:
                raise EmbeddingError(f"Embedding chunk {i}: {e}") from e
        return np.array(vecs, dtype="float32")

    # Parallel batching mode (Rank 10 optimization)
    logger.info(f"[Rank 10] Embedding {total} texts with {EMB_MAX_WORKERS} workers")
    results = [None] * total  # Pre-allocate to maintain order
    completed = 0

    try:
        with ThreadPoolExecutor(max_workers=EMB_MAX_WORKERS) as executor:
            # Submit all embedding tasks (Rank 5 fix: pass retries instead of shared session)
            futures = {
                executor.submit(_embed_single_text, i, text, retries, total): i
                for i, text in enumerate(texts)
            }

            # Collect results as they complete
            for future in as_completed(futures):
                idx, emb = future.result()  # Will raise if _embed_single_text raised
                results[idx] = emb
                completed += 1

                # Log progress every 100 completions
                if completed % 100 == 0 or completed == total:
                    logger.info(f"  [{completed}/{total}]")

    except Exception as e:
        # If batching fails, log and re-raise
        logger.error(f"[Rank 10] Batched embedding failed: {e}")
        raise

    # Verify all results collected
    if any(r is None for r in results):
        missing = [i for i, r in enumerate(results) if r is None]
        raise EmbeddingError(f"Missing embeddings for indices: {missing}")

    return np.array(results, dtype="float32")


def load_embedding_cache() -> dict:
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


def save_embedding_cache(cache: dict):
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


def embed_query(question: str, retries=0) -> np.ndarray:
    """Embed a single query using configured backend with optional caching."""
    # Note: This is a simplified version. Full implementation with caching
    # would be in the retrieval module
    if EMB_BACKEND == "local":
        vec = embed_local_batch([question], normalize=True)
        return vec[0]
    else:
        vecs = embed_texts([question], retries=retries)
        # Normalize for cosine similarity
        vec = vecs[0]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
