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
    is_rtf
)
from clockify_rag.answer import (
    apply_mmr_diversification,
    apply_reranking,
    extract_citations,
    validate_citations,
    generate_llm_answer,
    answer_once
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
from clockify_rag.cli import ensure_index_ready, chat_repl, warmup_on_startup
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

# Rank 23: NLTK for sentence-aware chunking (with optional download control)
_NLTK_AVAILABLE = False
_NLTK_DOWNLOAD_ATTEMPTED = False

def _ensure_nltk(auto_download=None):
    """Ensure NLTK is available, with optional download control for offline environments."""
    global _NLTK_AVAILABLE, _NLTK_DOWNLOAD_ATTEMPTED

    if _NLTK_AVAILABLE:
        return True

    try:
        import nltk
    except ImportError:
        logger.warning("NLTK not installed. Chunking will use simpler fallback.")
        return False

    # Check if we already have punkt
    try:
        nltk.data.find('tokenizers/punkt')
        _NLTK_AVAILABLE = True
        return True
    except LookupError:
        pass

    # Determine if we should download
    if auto_download is None:
        # Check environment variable (default: allow download unless explicitly disabled)
        auto_download = os.environ.get("NLTK_AUTO_DOWNLOAD", "1").lower() not in {"0", "false", "no", "off"}

    if auto_download and not _NLTK_DOWNLOAD_ATTEMPTED:
        _NLTK_DOWNLOAD_ATTEMPTED = True
        logger.info("Downloading NLTK punkt tokenizer (one-time setup)...")
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)  # For newer NLTK versions
            _NLTK_AVAILABLE = True
            logger.info("NLTK punkt downloaded successfully.")
            return True
        except Exception as e:
            logger.warning(f"NLTK download failed ({e}). Using simpler chunking fallback.")
            return False
    else:
        if not auto_download:
            logger.warning("NLTK auto-download disabled (NLTK_AUTO_DOWNLOAD=0). Using simpler chunking fallback.")
        return False

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


# ====== LOCAL EMBEDDINGS (v4.1 - Section 2) ======
_ST_ENCODER = None
_ST_BATCH_SIZE = 96

def _load_st_encoder():
    """Lazy-load SentenceTransformer model once."""
    global _ST_ENCODER
    if _ST_ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _ST_ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
        logger.debug("Loaded SentenceTransformer: all-MiniLM-L6-v2 (384-dim)")
    return _ST_ENCODER


# ====== FAISS ANN INDEX (v4.1 - Section 3) ======
_FAISS_INDEX = None
_FAISS_LOCK = threading.Lock()
# Stores profiling data from the most recent `retrieve` invocation.
RETRIEVE_PROFILE_LAST = {}

def _try_load_faiss():
    """Try importing FAISS; returns None if not available."""
    try:
        import faiss
        return faiss
    except ImportError:
        logger.info("info: ann=fallback reason=missing-faiss")
        return None




# ====== HYBRID SCORING (v4.1 - Section 4) ======

def hybrid_score(bm25_score: float, dense_score: float, alpha: float = 0.5) -> float:
    """Blend BM25 and dense scores: alpha * bm25_norm + (1 - alpha) * dense_norm."""
    return alpha * bm25_score + (1 - alpha) * dense_score

# ====== DYNAMIC PACKING (v4.1 - Section 5) ======
def pack_snippets_dynamic(chunk_ids: list, chunks: dict, budget_tokens: int | None = None, target_util: float = 0.75) -> tuple:
    """Pack snippets with dynamic targeting. Returns (snippets, used_tokens, was_truncated)."""
    if budget_tokens is None:
        budget_tokens = config.CTX_TOKEN_BUDGET
    if not chunk_ids:
        return [], 0, False

    snippets: list[str] = []
    token_count = 0
    target = int(budget_tokens * target_util)

    for cid in chunk_ids:
        try:
            chunk = chunks[cid]
            snippet_tokens = max(1, len(chunk.get("text", "")) // 4)
            separator_tokens = 16
            new_total = token_count + snippet_tokens + separator_tokens

            if new_total > budget_tokens:
                if snippets:
                    return snippets + [{"id": "[TRUNCATED]", "text": "..."}], token_count, True
                else:
                    snippets.append(chunk)
                    return snippets, token_count + snippet_tokens, True

            snippets.append(chunk)
            token_count = new_total

            if token_count >= target:
                break
        except (KeyError, IndexError, AttributeError, TypeError) as e:
            # Skip chunks with invalid data or missing indices
            logger.debug(f"Skipping chunk {cid}: {e}")
            continue

    return snippets, token_count, False

# ====== KPI LOGGING (v4.1 - Section 6) ======

# ====== JSON OUTPUT (v4.1 - Section 9) ======
def answer_to_json(answer: str, citations: list, used_tokens: int | None, topk: int, packed: int, confidence: int | None = None) -> dict:
    """Convert answer and metadata to JSON structure.

    Args:
        answer: Generated answer text.
        citations: Sequence of citation identifiers (chunk IDs).
        used_tokens: Actual token budget consumed when packing context.
        topk: Retrieval depth requested.
        packed: Maximum number of snippets packed.
        confidence: LLM confidence score (0-100), if available.
    """
    budget_tokens = 0 if used_tokens is None else int(used_tokens)
    result = {
        "answer": answer,
        "citations": citations,
        "debug": {
            "meta": {
                "used_tokens": budget_tokens,
                "topk": topk,
                "packed": packed,
                "emb_backend": config.EMB_BACKEND,
                "ann": config.USE_ANN,
                "alpha": config.ALPHA_HYBRID
            },
            "timing": {
                "retrieve_ms": KPI.retrieve_ms,
                "ann_ms": KPI.ann_ms,
                "rerank_ms": KPI.rerank_ms,
                "ask_ms": KPI.ask_ms,
                "total_ms": KPI.retrieve_ms + KPI.rerank_ms + KPI.ask_ms
            }
        }
    }

    # Include confidence if available
    if confidence is not None:
        result["confidence"] = confidence

    return result

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

# ====== POLICY GUARDRAILS ======
def looks_sensitive(question: str) -> bool:
    """Check if question involves sensitive intent (account/billing/PII)."""
    sensitive_keywords = {
        # Financial
        "invoice", "billing", "credit card", "payment", "salary", "account balance",
        # Authentication & Secrets
        "password", "token", "api key", "secret", "private key",
        # PII
        "ssn", "social security", "iban", "swift", "routing number", "account number",
        "phone number", "email address", "home address", "date of birth",
        # Compliance
        "gdpr", "pii", "personally identifiable", "personal data"
    }
    q_lower = question.lower()
    return any(kw in q_lower for kw in sensitive_keywords)

def inject_policy_preamble(snippets_block: str, question: str) -> str:
    """Optionally prepend policy reminder for sensitive queries."""
    if looks_sensitive(question):
        policy = "[INTERNAL POLICY]\nDo not reveal PII, account secrets, or payment details. For account changes, redirect to secure internal admin panel.\n\n"
        return policy + snippets_block
    return snippets_block

# ====== INPUT SANITIZATION ======
def sanitize_question(q: str, max_length: int = 2000) -> str:
    """Validate and sanitize user question.

    Args:
        q: User question string
        max_length: Maximum allowed question length (default: 2000)

    Returns:
        Sanitized question string

    Raises:
        ValueError: If question is invalid (empty, too long, invalid characters)
    """
    # Type check
    if not isinstance(q, str):
        raise ValueError(f"Question must be a string, got {type(q).__name__}")

    # Strip whitespace
    q = q.strip()

    # Check length
    if len(q) == 0:
        raise ValueError("Question cannot be empty. Hint: Provide a meaningful question about Clockify.")
    if len(q) > max_length:
        raise ValueError(
            f"Question too long (max {max_length} characters, got {len(q)}).\n"
            f"Hint: Break your question into smaller, focused queries."
        )

    # Check for null bytes first (specific check)
    if '\x00' in q:
        raise ValueError("Question contains control characters")

    # Check for control characters (except newline, tab, carriage return)
    if any(ord(c) < 32 and c not in '\n\r\t' for c in q):
        raise ValueError("Question contains invalid control characters")

    # Check for suspicious patterns (basic prompt injection detection)
    suspicious_patterns = [
        '<script',
        'javascript:',
        'eval(',
        'exec(',
        '__import__',
        '<?php',
    ]
    q_lower = q.lower()
    for pattern in suspicious_patterns:
        if pattern in q_lower:
            raise ValueError(f"Question contains suspicious pattern: {pattern}")

    return q

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
    # v4.1: Declare globals at function start (Section 7)
    # Note: Most config now in clockify_rag.config module (no global declaration needed)
    global QUERY_LOG_DISABLED

    # Create parent parser for common flags shared across subcommands
    common_flags = argparse.ArgumentParser(add_help=False)
    common_flags.add_argument("--emb-backend", choices=["local", "ollama"], default=config.EMB_BACKEND,
                             help="Embedding backend: local (SentenceTransformer) or ollama (default local)")
    common_flags.add_argument("--ann", choices=["faiss", "none"], default=config.USE_ANN,
                             help="ANN index: faiss (IVFFlat) or none (full-scan, default faiss)")
    common_flags.add_argument("--alpha", type=float, default=config.ALPHA_HYBRID,
                             help="Hybrid scoring blend: alpha*BM25 + (1-alpha)*dense (default 0.5)")

    ap = argparse.ArgumentParser(
        prog="clockify_support_cli",
        description="Clockify internal support chatbot (offline, stateless, closed-book)"
    )

    # Global logging and config arguments
    ap.add_argument("--log", default="INFO", choices=["DEBUG", "INFO", "WARN"],
                    help="Logging level (default INFO)")
    ap.add_argument("--no-log", action="store_true",
                    help="Disable query log file writes (privacy mode)")
    ap.add_argument("--ollama-url", type=str, default=None,
                    help="Ollama endpoint (default from config.OLLAMA_URL env or http://127.0.0.1:11434)")
    ap.add_argument("--gen-model", type=str, default=None,
                    help="Generation model name (default from config.GEN_MODEL env or qwen2.5:32b)")
    ap.add_argument("--emb-model", type=str, default=None,
                    help="Embedding model name (default from config.EMB_MODEL env or nomic-embed-text)")
    ap.add_argument("--ctx-budget", type=int, default=None,
                    help="Context token budget (default from CTX_BUDGET env or 6000)")
    ap.add_argument("--query-expansions", type=str, default=None,
                    help="Path to JSON query expansion overrides (default config/query_expansions.json or CLOCKIFY_QUERY_EXPANSIONS env)")
    # Global-only flags
    ap.add_argument("--selftest", action="store_true", help="Run self-tests and exit (v4.1)")
    ap.add_argument("--profile", action="store_true", help="Enable cProfile performance profiling (Rank 29)")

    subparsers = ap.add_subparsers(dest="cmd")

    # Build subparser with common flags
    b = subparsers.add_parser("build", help="Build knowledge base", parents=[common_flags])
    b.add_argument("md_path", help="Path to knowledge_full.md")
    b.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retries for transient errors (default 2)")

    # Chat subparser with common flags
    c = subparsers.add_parser("chat", help="Start REPL", parents=[common_flags])
    c.add_argument("--debug", action="store_true", help="Print retrieval diagnostics")
    c.add_argument("--rerank", action="store_true", help="Enable LLM-based reranking")
    c.add_argument("--topk", type=int, default=config.DEFAULT_TOP_K, help="Top-K candidates (default 12)")
    c.add_argument("--pack", type=int, default=config.DEFAULT_PACK_TOP, help="Snippets to pack (default 6)")
    c.add_argument("--threshold", type=float, default=config.DEFAULT_THRESHOLD, help="Cosine threshold (default 0.30)")
    c.add_argument("--seed", type=int, default=config.DEFAULT_SEED, help="Random seed for LLM (default 42)")
    c.add_argument("--num-ctx", type=int, default=config.DEFAULT_NUM_CTX, help=f"LLM context window (default {config.DEFAULT_NUM_CTX})")
    c.add_argument("--num-predict", type=int, default=config.DEFAULT_NUM_PREDICT, help="LLM max generation tokens (default 512)")
    c.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retries for transient errors (default 2)")
    c.add_argument("--det-check", action="store_true", help="Determinism check: ask same Q twice, compare hashes")
    c.add_argument("--no-expand", action="store_true",
                   help="Disable query expansion (synonym substitution)")
    c.add_argument("--faiss-multiplier", type=int, default=config.FAISS_CANDIDATE_MULTIPLIER,
                   help="FAISS candidate multiplier: retrieve top_k * N for reranking (default 3)")
    c.add_argument("--json", action="store_true", help="Output answer as JSON with metrics (v4.1)")

    # Ask subparser with common flags
    a = subparsers.add_parser("ask", help="Answer a single question and exit", parents=[common_flags])
    a.add_argument("question", help="Question to answer")
    a.add_argument("--debug", action="store_true", help="Print retrieval diagnostics")
    a.add_argument("--rerank", action="store_true", help="Enable LLM-based reranking")
    a.add_argument("--topk", type=int, default=config.DEFAULT_TOP_K, help="Top-K candidates (default 12)")
    a.add_argument("--pack", type=int, default=config.DEFAULT_PACK_TOP, help="Snippets to pack (default 6)")
    a.add_argument("--threshold", type=float, default=config.DEFAULT_THRESHOLD, help="Cosine threshold (default 0.30)")
    a.add_argument("--seed", type=int, default=config.DEFAULT_SEED, help="Random seed for LLM (default 42)")
    a.add_argument("--num-ctx", type=int, default=config.DEFAULT_NUM_CTX, help=f"LLM context window (default {config.DEFAULT_NUM_CTX})")
    a.add_argument("--num-predict", type=int, default=config.DEFAULT_NUM_PREDICT, help="LLM max generation tokens (default 512)")
    a.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retries for transient errors (default 2)")
    a.add_argument("--no-expand", action="store_true",
                   help="Disable query expansion (synonym substitution)")
    a.add_argument("--faiss-multiplier", type=int, default=config.FAISS_CANDIDATE_MULTIPLIER,
                   help="FAISS candidate multiplier: retrieve top_k * N for reranking (default 3)")
    a.add_argument("--json", action="store_true", help="Output answer as JSON with metrics (v4.1)")

    args = ap.parse_args()

    # Setup logging after CLI arg parsing
    level = getattr(logging, args.log if hasattr(args, "log") else "INFO")
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    # Configure query expansion dictionary overrides
    if getattr(args, "query_expansions", None):
        set_query_expansion_path(args.query_expansions)
    elif os.environ.get(QUERY_EXPANSIONS_ENV_VAR):
        set_query_expansion_path(os.environ[QUERY_EXPANSIONS_ENV_VAR])
    else:
        set_query_expansion_path(None)

    try:
        load_query_expansion_dict(force_reload=True, suppress_errors=False)
    except ValueError as exc:
        logger.error("CONFIG ERROR: %s", exc)
        sys.exit(1)
    if getattr(args, "no_log", False):
        QUERY_LOG_DISABLED = True

    # v4.1: Update globals from CLI args (Section 7)
    config.EMB_BACKEND = args.emb_backend
    config.USE_ANN = args.ann
    config.ALPHA_HYBRID = args.alpha

    # Update FAISS multiplier if provided in subcommand args
    if hasattr(args, "faiss_multiplier"):
        config.FAISS_CANDIDATE_MULTIPLIER = args.faiss_multiplier

    # v4.1: Run selftest if requested (Section 8)
    if getattr(args, "selftest", False):
        success = run_selftest()
        sys.exit(0 if success else 1)

    # Validate and set config from CLI args
    try:
        validate_and_set_config(
            ollama_url=args.ollama_url,
            gen_model=args.gen_model,
            emb_model=args.emb_model,
            ctx_budget=args.ctx_budget
        )
        validate_chunk_config()
        check_pytorch_mps()  # v4.1.2: Check MPS availability on M1 Macs
    except ValueError as e:
        logger.error(f"CONFIG ERROR: {e}")
        sys.exit(1)

    # Auto-start REPL if no command given
    if args.cmd is None:
        chat_repl()
        return

    if args.cmd == "build":
        build(args.md_path, retries=getattr(args, "retries", 0))
        return

    if args.cmd == "ask":
        _log_config_summary(
            use_rerank=args.rerank,
            pack_top=args.pack,
            seed=args.seed,
            threshold=args.threshold,
            top_k=args.topk,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            retries=getattr(args, "retries", 0)
        )
        chunks, vecs_n, bm, hnsw = ensure_index_ready(retries=getattr(args, "retries", 0))
        ans, meta = answer_once(
            args.question,
            chunks,
            vecs_n,
            bm,
            top_k=args.topk,
            pack_top=args.pack,
            threshold=args.threshold,
            use_rerank=args.rerank,
            debug=args.debug,
            hnsw=hnsw,
            seed=args.seed,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            retries=getattr(args, "retries", 0)
        )
        if getattr(args, "json", False):
            used_tokens = meta.get("used_tokens")
            if used_tokens is None:
                used_tokens = len(meta.get("selected", []))
            output = answer_to_json(
                ans,
                meta.get("selected", []),
                used_tokens,
                args.topk,
                args.pack,
                meta.get("confidence")
            )
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            print(ans)
        return

    if args.cmd == "chat":
        # Task A: Determinism check
        if getattr(args, "det_check", False):
            # Load index once for determinism test
            for fname in [config.FILES["chunks"], config.FILES["emb"], config.FILES["meta"], config.FILES["bm25"], config.FILES["index_meta"]]:
                if not os.path.exists(fname):
                    logger.info("[rebuild] artifacts missing for det-check: building...")
                    if os.path.exists("knowledge_full.md"):
                        build("knowledge_full.md", retries=getattr(args, "retries", 0))
                    break
            result = load_index()
            if result:
                # v4.1: Determinism check using Ollama with tuple timeouts and retry helper
                try:
                    seed = 42
                    np.random.seed(seed)
                    prompt = "What is Clockify?"
                    payload = {"model": config.GEN_MODEL, "prompt": prompt, "options": {"seed": seed}}

                    r1 = http_post_with_retries(f"{config.OLLAMA_URL}/api/generate", payload,
                                                retries=2, timeout=(CHAT_CONNECT_T, config.CHAT_READ_T))
                    ans1 = r1.json().get("response", "")

                    np.random.seed(seed)
                    r2 = http_post_with_retries(f"{config.OLLAMA_URL}/api/generate", payload,
                                                retries=2, timeout=(CHAT_CONNECT_T, config.CHAT_READ_T))
                    ans2 = r2.json().get("response", "")

                    h1 = hashlib.md5(ans1.encode()).hexdigest()[:16]
                    h2 = hashlib.md5(ans2.encode()).hexdigest()[:16]
                    deterministic = (h1 == h2)
                    logger.info(f"[DETERMINISM] run1={h1} run2={h2} deterministic={deterministic}")
                    print(f'[DETERMINISM] run1={h1} run2={h2} deterministic={"true" if deterministic else "false"}')
                    sys.exit(0 if deterministic else 1)
                except Exception as e:
                    logger.error(f"❌ Determinism test failed: {e}")
                    sys.exit(1)
            else:
                logger.error("failed to load index for det-check")
                sys.exit(1)

        # Normal chat REPL
        chat_repl(
            top_k=args.topk,
            pack_top=args.pack,
            threshold=args.threshold,
            use_rerank=args.rerank,
            debug=args.debug,
            seed=args.seed,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            retries=getattr(args, "retries", 0),
            use_json=getattr(args, "json", False)  # v4.1: JSON output flag
        )
        return

    if args.cmd == "ask":
        chunks, vecs_n, bm, hnsw = ensure_index_ready(retries=getattr(args, "retries", 0))

        _log_config_summary(
            use_rerank=args.rerank,
            pack_top=args.pack,
            seed=args.seed,
            threshold=args.threshold,
            top_k=args.topk,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            retries=getattr(args, "retries", 0)
        )

        warmup_on_startup()

        ans, meta = answer_once(
            args.question,
            chunks,
            vecs_n,
            bm,
            top_k=args.topk,
            pack_top=args.pack,
            threshold=args.threshold,
            use_rerank=args.rerank,
            debug=args.debug,
            hnsw=hnsw,
            seed=args.seed,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            retries=getattr(args, "retries", 0)
        )

        if getattr(args, "json", False):
            used_tokens = meta.get("used_tokens")
            if used_tokens is None:
                used_tokens = len(meta.get("selected", []))
            output = answer_to_json(
                ans,
                meta.get("selected", []),
                used_tokens,
                args.topk,
                args.pack,
                meta.get("confidence")
            )
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            print(ans)
        return

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
