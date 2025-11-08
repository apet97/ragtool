#!/usr/bin/env python3
"""
Clockify Internal Support CLI – Stateless RAG with Hybrid Retrieval

HOW TO RUN
==========
  # Build knowledge base (one-time)
  python3 clockify_support_cli_final.py build knowledge_full.md

  # Start interactive REPL
  python3 clockify_support_cli_final.py chat [--debug] [--rerank] [--topk 12] [--pack 6] [--threshold 0.30]

  # Or auto-start REPL with no args
  python3 clockify_support_cli_final.py

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

# Import all configuration and core functionality from package
import clockify_rag.config as config
from clockify_rag.config import KPI, LOG_QUERY_INCLUDE_CHUNKS, QUERY_LOG_FILE, LOG_QUERY_INCLUDE_ANSWER, LOG_QUERY_ANSWER_PLACEHOLDER
from clockify_rag.caching import QueryCache, RateLimiter, get_query_cache, get_rate_limiter
from clockify_rag.chunking import build_chunks, sliding_chunks
from clockify_rag.embedding import embed_texts, load_embedding_cache, save_embedding_cache, validate_ollama_embeddings, embed_local_batch
from clockify_rag.indexing import build, load_index, build_bm25, bm25_scores, build_faiss_index, save_faiss_index
from clockify_rag.retrieval import (
    expand_query,
    embed_query,
    retrieve,
    pack_snippets,
    coverage_ok,
    ask_llm,
    count_tokens,
    tokenize,
    normalize_scores_zscore,
    DenseScoreStore,
    QUERY_EXPANSIONS_ENV_VAR,
    load_query_expansion_dict,
    reset_query_expansion_cache,
    set_query_expansion_path,
    hybrid_score,
    pack_snippets_dynamic,
)
from clockify_rag.metrics import get_metrics, time_operation
from clockify_rag.utils import (
    validate_ollama_url,
    compute_sha256,
    build_lock,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_save_npy,
    _fsync_dir,
    norm_ws,
    strip_noise,
    _release_lock_if_owner,
    _pid_alive,
    validate_and_set_config,
    validate_chunk_config,
    check_pytorch_mps,
    _log_config_summary,
    log_kpi,
    log_event,
    atomic_write_bytes,
    atomic_write_text,
    is_rtf,
    _ensure_nltk,
    _load_st_encoder,
    _try_load_faiss,
    looks_sensitive,
    inject_policy_preamble,
    sanitize_question
)
from clockify_rag.answer import (
    apply_mmr_diversification,
    apply_reranking,
    extract_citations,
    validate_citations,
    generate_llm_answer,
    answer_once,
    answer_to_json
)
from clockify_rag.http_utils import (
    get_session,
    http_post_with_retries,
    _mount_retries
)
from clockify_rag.retrieval import (
    rerank_with_llm,
    truncate_to_token_budget
)
from clockify_rag.caching import log_query
from clockify_rag.cli import (
    ensure_index_ready,
    chat_repl,
    warmup_on_startup,
    setup_cli_args,
    configure_logging_and_config,
    handle_build_command,
    handle_ask_command,
    handle_chat_command
)
from clockify_rag.indexing import load_faiss_index

# Re-export config constants for backward compatibility with tests
# Tests import these directly from this module
DEFAULT_TOP_K = config.DEFAULT_TOP_K
DEFAULT_PACK_TOP = config.DEFAULT_PACK_TOP
DEFAULT_THRESHOLD = config.DEFAULT_THRESHOLD
DEFAULT_SEED = config.DEFAULT_SEED
DEFAULT_NUM_CTX = config.DEFAULT_NUM_CTX
DEFAULT_NUM_PREDICT = config.DEFAULT_NUM_PREDICT
DEFAULT_RETRIES = config.DEFAULT_RETRIES
GEN_MODEL = config.GEN_MODEL
EMB_MODEL = config.EMB_MODEL
EMB_BACKEND = config.EMB_BACKEND
OLLAMA_URL = config.OLLAMA_URL
REFUSAL_STR = config.REFUSAL_STR
CTX_TOKEN_BUDGET = config.CTX_TOKEN_BUDGET
CHUNK_CHARS = config.CHUNK_CHARS
CHUNK_OVERLAP = config.CHUNK_OVERLAP
FILES = config.FILES
# Timeout constants (Task G)
EMB_CONNECT_T = config.EMB_CONNECT_T
EMB_READ_T = config.EMB_READ_T
CHAT_CONNECT_T = config.CHAT_CONNECT_T
CHAT_READ_T = config.CHAT_READ_T

# ====== MODULE LOGGER ======
logger = logging.getLogger(__name__)

# ====== IMPORT EXCEPTIONS FROM PACKAGE ======
from clockify_rag.exceptions import EmbeddingError, LLMError, IndexLoadError, BuildError

# ====== CLI-SPECIFIC CONFIG (inherits from package) ======
# All core configuration now imported from clockify_rag.config
# Reference as: config.OLLAMA_URL, config.GEN_MODEL, etc.

# CLI-specific file paths and flags (not in package config)
BUILD_LOCK = ".build.lock"
BUILD_LOCK_TTL_SEC = int(os.environ.get("BUILD_LOCK_TTL_SEC", "900"))  # Task D: 15 minutes default
QUERY_LOG_DISABLED = False  # Can be set to True via --no-log flag

# Note: Query expansion now handled by clockify_rag.retrieval.expand_query()

# ====== CLEANUP HANDLERS ======

atexit.register(_release_lock_if_owner)

# Global requests session for keep-alive and retry logic
REQUESTS_SESSION = None
REQUESTS_SESSION_RETRIES = 0



# v4.1: HTTP POST helper with retry logic


# ====== FAISS ANN INDEX (v4.1 - Section 3) ======
_FAISS_INDEX = None
_FAISS_LOCK = threading.Lock()
# Stores profiling data from the most recent `retrieve` invocation.
RETRIEVE_PROFILE_LAST = {}




# ====== KPI LOGGING (v4.1 - Section 6) ======

# ====== SELF-TEST INTEGRATION CHECKS (v4.1 - Section 8) ======
# Note: Detailed unit tests are in test_* functions below.
# These are integration/smoke tests that verify key components.



# ====== CONFIG VALIDATION ======





# ====== SYSTEM PROMPT (Rank 25: Few-shot examples added) ======
SYSTEM_PROMPT = f"""You are CAKE.com Internal Support for Clockify.
Closed-book. Only use SNIPPETS. If info is missing, reply exactly "{config.REFUSAL_STR}" and set confidence to 0.
Respond with a single JSON object that matches this schema:
{{
  "answer": "<complete response>",
  "confidence": <0-100 integer>
}}
Guidelines for the answer field:
- Use the user's language.
- Be precise. No speculation. No external info. No web search.
- Include the following sections in order inside the answer text (you may format them with numbered or bulleted lists):
  1. Direct answer.
  2. Steps.
  3. Notes by role/plan/region if relevant.
  4. Citations with snippet IDs like [id1, id2], including URLs inline if present.
- If SNIPPETS disagree, explain the conflict and provide the safest interpretation.
- Ensure the entire output remains valid JSON with no extra prose or markdown wrappers.

Now answer the user's question."""

USER_WRAPPER = """SNIPPETS:
{snips}

QUESTION:
{q}

Respond with only a JSON object following the schema {{"answer": "...", "confidence": 0-100}}.
Keep all narrative content inside the answer field and include citations as described in the system message.
Do not add markdown fences or text outside the JSON object."""

RERANK_PROMPT = """You rank passages for a Clockify support answer. Score each 0.0–1.0 strictly.
Output JSON only: [{"id":"<chunk_id>","score":0.82}, ...].

QUESTION:
{q}

PASSAGES:
{passages}"""

# ====== UTILITIES ======










# ====== REMOVED REDUNDANT IMPLEMENTATIONS (2025-11-08) ======
# The following functions were removed as they duplicate library imports:
# - tokenize, count_tokens → imported from clockify_rag.retrieval (line 67)
# - compute_sha256 → imported from clockify_rag.utils (line 74)
# All functionality now uses the optimized library versions with proper caching,
# parallel processing, and dimension validation.
# ============================================================




# ====== REMOVED REDUNDANT IMPLEMENTATIONS (2025-11-08) ======
# The following KB parsing, embedding, and retrieval functions were removed
# as they duplicate library imports from clockify_rag package (lines 56-70):
#
# KB PARSING (imported from clockify_rag.chunking at line 56):
# - parse_articles, split_by_headings, sliding_chunks, build_chunks
#
# EMBEDDINGS (imported from clockify_rag.embedding at line 57):
# - validate_ollama_embeddings, load_embedding_cache, save_embedding_cache, embed_texts
#
# INDEXING (imported from clockify_rag.indexing at line 58):
# - build_bm25, bm25_scores
#
# RETRIEVAL (imported from clockify_rag.retrieval at lines 59-70):
# - normalize_scores_zscore, expand_query, embed_query, retrieve
#
# All functionality now uses the optimized library versions with:
# - Parallel embedding processing (3-5x faster)
# - Proper dimension validation
# - Cache metadata tracking
# - Thread-safe operations
# ============================================================


# ====== REMOVED REDUNDANT IMPLEMENTATIONS (2025-11-08) ======
# The following helper and indexing functions were removed as they duplicate
# library imports from clockify_rag package (lines 58-65):
#
# SNIPPET FORMATTING & PACKING (imported from clockify_rag.retrieval at line 63):
# - _fmt_snippet_header, pack_snippets
#
# COVERAGE & LLM (imported from clockify_rag.retrieval at lines 64-65):
# - coverage_ok, ask_llm
#
# INDEX BUILD & LOAD (imported from clockify_rag.indexing at line 58):
# - build, load_index
#
# All functionality now uses the optimized library versions.
# ============================================================

# ====== RATE LIMITING & QUERY CACHING ======
# Priority #2: Reuse package implementations instead of duplicate definitions (ROI 9/10)
# Classes QueryCache and RateLimiter are imported from clockify_rag.caching (line 54)
# Global instances use factory functions to ensure proper initialization

# Global rate limiter (10 queries per minute by default)
RATE_LIMITER = get_rate_limiter()

# Global query cache (100 entries, 1 hour TTL by default)
QUERY_CACHE = get_query_cache()

# ====== STRUCTURED LOGGING ======

# ====== ANSWER PIPELINE HELPERS ======






# ====== ANSWER (STATELESS) ======

def main():
    """Main entry point - delegates to CLI module for all functionality."""
    global QUERY_LOG_DISABLED

    # Parse command line arguments
    args = setup_cli_args()

    # Configure logging and global config
    QUERY_LOG_DISABLED = configure_logging_and_config(args)

    # Handle selftest if requested
    if getattr(args, "selftest", False):
        success = run_selftest()
        sys.exit(0 if success else 1)

    # Auto-start REPL if no command given
    if args.cmd is None:
        chat_repl()
        return

    # Route to appropriate command handler
    if args.cmd == "build":
        handle_build_command(args)
    elif args.cmd == "ask":
        handle_ask_command(args)
    elif args.cmd == "chat":
        handle_chat_command(args)

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
