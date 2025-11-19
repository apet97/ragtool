"""Remote embeddings client using LangChain OllamaEmbeddings.

This module provides remote-only embedding generation via the corporate
Ollama instance. Designed for:

- VPN environments (remote-first, no local models)
- Timeout safety (5s connection, 60s read timeout)
- Clean abstraction using LangChain's OllamaEmbeddings
- Fallback to legacy api_client if needed (deprecated)
"""

import logging
from typing import List

import numpy as np

# Use langchain-ollama (newer package, better maintained)
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    # Fallback to langchain-community for older installations
    from langchain_community.embeddings import OllamaEmbeddings  # type: ignore

from .config import RAG_OLLAMA_URL, RAG_EMBED_MODEL, EMB_CONNECT_T, EMB_READ_T

logger = logging.getLogger(__name__)

# Global instance (lazy-loaded)
_EMBEDDING_CLIENT: OllamaEmbeddings | None = None
_EMBEDDING_DIM: int | None = None


def get_embedding_client() -> OllamaEmbeddings:
    """Get or create the remote Ollama embeddings client (lazy-loaded).

    On first initialization, probes the embedding dimension by embedding a test string.

    Returns:
        OllamaEmbeddings instance connected to corporate Ollama server

    Example:
        ```python
        from clockify_rag.embeddings_client import get_embedding_client, embed_texts

        client = get_embedding_client()
        vecs = embed_texts(["Hello world", "Another text"])
        print(vecs.shape)  # (2, 768)
        ```
    """
    global _EMBEDDING_CLIENT, _EMBEDDING_DIM
    if _EMBEDDING_CLIENT is None:
        logger.debug(
            f"Initializing OllamaEmbeddings: model={RAG_EMBED_MODEL}, "
            f"base_url={RAG_OLLAMA_URL}, "
            f"timeouts=({EMB_CONNECT_T}s, {EMB_READ_T}s)"
        )
        # langchain-ollama OllamaEmbeddings accepts timeout via sync_client_kwargs
        _EMBEDDING_CLIENT = OllamaEmbeddings(
            base_url=RAG_OLLAMA_URL,
            model=RAG_EMBED_MODEL,
            # Pass timeout via sync_client_kwargs for httpx client
            # Tuple: (connect_timeout, read_timeout) in seconds
            sync_client_kwargs={"timeout": (EMB_CONNECT_T, EMB_READ_T)},
        )
        # Probe dimension once
        try:
            probe_vec = _EMBEDDING_CLIENT.embed_query("probe dimension")
            _EMBEDDING_DIM = len(probe_vec)
            logger.info(f"Remote embeddings initialized: {RAG_EMBED_MODEL} ({_EMBEDDING_DIM}-dim)")
        except Exception as e:
            logger.warning(f"Failed to probe embedding dimension: {e}; assuming 768-dim")
            _EMBEDDING_DIM = 768
    return _EMBEDDING_CLIENT


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed multiple texts using remote Ollama with L2 normalization.

    Args:
        texts: List of strings to embed

    Returns:
        NumPy array of shape (len(texts), embedding_dim) with float32 dtype, L2-normalized

    Raises:
        requests.Timeout: If Ollama is unreachable or too slow
        ValueError: If texts is empty
    """
    if not texts:
        # Return empty array with correct embedding dimension
        dim = _EMBEDDING_DIM or 768
        return np.zeros((0, dim), dtype=np.float32)

    logger.debug(f"Embedding {len(texts)} texts via remote Ollama")
    client = get_embedding_client()

    # LangChain's OllamaEmbeddings.embed_documents handles batching internally
    try:
        embeddings_list = client.embed_documents(texts)
        # Convert to NumPy array (float32 for memory efficiency)
        embeddings_array = np.array(embeddings_list, dtype=np.float32)

        # Normalize rows to unit length (L2 normalization for cosine similarity)
        # This ensures retrieval uses cosine similarity correctly
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True) + 1e-9
        embeddings_array = embeddings_array / norms

        logger.debug(f"Successfully embedded and normalized {len(texts)} texts: shape {embeddings_array.shape}")
        return embeddings_array
    except Exception as e:
        logger.error(f"Failed to embed texts: {e}")
        raise


def embed_query(text: str) -> np.ndarray:
    """Embed a single query text using remote Ollama with L2 normalization.

    Args:
        text: Query string to embed

    Returns:
        1D NumPy array of shape (embedding_dim,) with float32 dtype, L2-normalized

    Raises:
        requests.Timeout: If Ollama is unreachable or too slow
        ValueError: If text is empty
    """
    if not text:
        raise ValueError("Cannot embed empty text")

    logger.debug(f"Embedding query via remote Ollama")
    client = get_embedding_client()

    try:
        # LangChain's OllamaEmbeddings.embed_query is optimized for single queries
        embedding_list = client.embed_query(text)
        embedding_array = np.array(embedding_list, dtype=np.float32)

        # Normalize to unit length (L2 normalization for cosine similarity)
        norm = np.linalg.norm(embedding_array) + 1e-9
        embedding_array = embedding_array / norm

        logger.debug(f"Successfully embedded and normalized query: shape {embedding_array.shape}")
        return embedding_array
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        raise


def clear_cache():
    """Clear the cached embedding client instance.

    Useful for testing or switching Ollama endpoints at runtime.
    """
    global _EMBEDDING_CLIENT
    _EMBEDDING_CLIENT = None
    logger.debug("Cleared embedding client cache")
