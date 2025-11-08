"""CLI and REPL functions for the Clockify RAG system.

This module contains the interactive REPL, warmup logic, and index loading helpers.
Extracted from clockify_support_cli_final.py to reduce file size and improve modularity.
"""

import hashlib
import json
import logging
import os
import sys
from typing import Tuple, Optional, Any

from . import config
from .indexing import build, load_index
from .utils import _log_config_summary
from .answer import answer_once
from .caching import get_query_cache

logger = logging.getLogger(__name__)


def ensure_index_ready(retries=0) -> Tuple:
    """Ensure retrieval artifacts are present and return loaded index components.

    Returns:
        Tuple of (chunks, vecs_n, bm, hnsw) for backward compatibility with CLI code.
        The library's load_index() returns a dict, which we unpack here.
    """

    artifacts_ok = True
    for fname in [config.FILES["chunks"], config.FILES["emb"], config.FILES["meta"], config.FILES["bm25"], config.FILES["index_meta"]]:
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

    # Handle both dict (from library) and tuple (from test mocks) for backward compatibility
    if isinstance(result, dict):
        # Library's load_index() returns a dictionary
        chunks = result["chunks"]
        vecs_n = result["vecs_n"]
        bm = result["bm"]
        hnsw = result.get("faiss_index")  # Note: was "hnsw" in old code, but library returns "faiss_index"
    elif isinstance(result, tuple):
        # Test mocks return a tuple (chunks, vecs_n, bm, hnsw)
        chunks, vecs_n, bm, hnsw = result
    else:
        raise TypeError(f"load_index() must return dict or tuple, got {type(result)}")

    return chunks, vecs_n, bm, hnsw


def chat_repl(top_k=12, pack_top=6, threshold=0.30, use_rerank=False, debug=False, seed=config.DEFAULT_SEED, num_ctx=config.DEFAULT_NUM_CTX, num_predict=config.DEFAULT_NUM_PREDICT, retries=0, use_json=False):
    """Stateless REPL loop - Task I. v4.1: JSON output support.

    OPTIMIZATION: Loads query cache from disk on startup and saves on exit for persistence.
    """
    # Task I: log config summary at startup
    _log_config_summary(use_rerank=use_rerank, pack_top=pack_top, seed=seed, threshold=threshold, top_k=top_k, num_ctx=num_ctx, num_predict=num_predict, retries=retries)

    # Lazy build and startup sanity check
    chunks, vecs_n, bm, hnsw = ensure_index_ready(retries=retries)

    # Rank 22: Persist query cache across REPL sessions
    query_cache = get_query_cache()
    query_cache.load()  # Load previous session's cache
    logger.info(f"Query cache loaded: {len(query_cache.cache)} entries")

    warmup_on_startup()

    print("━" * 60)
    print("Clockify Support CLI – Type ':exit' to quit, ':debug' to toggle diagnostics")
    print("━" * 60)

    debug_enabled = debug  # Local toggle

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not question:
            continue

        # REPL commands
        if question == ":exit":
            break
        if question == ":debug":
            debug_enabled = not debug_enabled
            print(f"Debug mode: {'ON' if debug_enabled else 'OFF'}")
            continue

        # Answer the question
        answer, meta = answer_once(
            question,
            chunks,
            vecs_n,
            bm,
            top_k=top_k,
            pack_top=pack_top,
            threshold=threshold,
            use_rerank=use_rerank,
            debug=debug_enabled,
            hnsw=hnsw,
            seed=seed,
            num_ctx=num_ctx,
            num_predict=num_predict,
            retries=retries
        )

        if use_json:
            # v4.1: JSON output mode
            used_tokens = meta.get("used_tokens")
            if used_tokens is None:
                used_tokens = len(meta.get("selected", []))
            output = {
                "answer": answer,
                "citations": meta.get("selected", []),
                "used_tokens": used_tokens,
                "topk": top_k,
                "packed": len(meta.get("selected", [])),
                "confidence": meta.get("confidence")
            }
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            print(f"\n{answer}")

        # Show debug info if enabled
        if debug_enabled and "selected" in meta:
            print(f"\n[DEBUG] Retrieved: {len(meta.get('selected', []))} chunks")
            print(f"[DEBUG] Scores: {meta.get('scores', [])[:5]}")

    # Save cache on exit
    query_cache.save()
    logger.info(f"Query cache saved: {len(query_cache.cache)} entries")


def warmup_on_startup():
    """Warm-up embeddings, LLM, and FAISS on startup (reduces first-token latency).

    OPTIMIZATION: Preloads FAISS index to eliminate 50-200ms penalty on first query.
    """
    warmup_enabled = os.environ.get("WARMUP", "1").lower() in ("1", "true", "yes")
    if not warmup_enabled:
        logger.debug("Warm-up disabled via WARMUP=0")
        return

    logger.info("Warming up...")

    # Warm-up embedding (Rank 3: 50-100ms saved on first query)
    try:
        from .embedding import embed_texts
        embed_texts(["warmup query"], suppress_errors=True)
    except Exception as e:
        logger.warning(f"Embedding warmup failed: {e}")

    # Warm-up LLM (Rank 3: 200-500ms saved on first query)
    try:
        from .http_utils import http_post_with_retries
        payload = {"model": config.GEN_MODEL, "prompt": "Hi", "options": {"num_predict": 1}}
        http_post_with_retries(
            f"{config.OLLAMA_URL}/api/generate",
            payload,
            retries=0,
            timeout=(3, 10)
        )
    except Exception as e:
        logger.warning(f"LLM warmup failed: {e}")

    # OPTIMIZATION: Preload FAISS index (if enabled)
    if config.USE_ANN == "faiss":
        try:
            from .indexing import load_faiss_index
            _ = load_faiss_index()
            logger.debug("FAISS index preloaded")
        except Exception as e:
            logger.warning(f"FAISS warmup failed: {e}")

    logger.info("Warmup complete")
