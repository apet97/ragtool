"""CLI and REPL functions for the Clockify RAG system.

This module contains the interactive REPL, warmup logic, and index loading helpers.
Extracted from clockify_support_cli_final.py to reduce file size and improve modularity.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import numpy as np
from typing import Tuple, Optional, Any, List

from . import config
from .indexing import build, load_index
from .utils import _log_config_summary, validate_and_set_config, validate_chunk_config, check_pytorch_mps
from .answer import answer_once, answer_to_json
from .caching import get_query_cache
from .retrieval import set_query_expansion_path, load_query_expansion_dict, QUERY_EXPANSIONS_ENV_VAR
from .precomputed_cache import get_precomputed_cache
from .api_client import get_llm_client, ChatCompletionOptions, ChatMessage

logger = logging.getLogger(__name__)


def ensure_index_ready(retries=0) -> Tuple:
    """Ensure retrieval artifacts are present and return loaded index components.

    Returns:
        Tuple of (chunks, vecs_n, bm, hnsw) for backward compatibility with CLI code.
        The library's load_index() returns a dict, which we unpack here.
    """
    from .error_handlers import log_and_raise
    from .exceptions import IndexLoadError

    artifacts_ok = True
    missing_files = []
    for fname in [config.FILES["chunks"], config.FILES["emb"], config.FILES["meta"], config.FILES["bm25"], config.FILES["index_meta"]]:
        if not os.path.exists(fname):
            artifacts_ok = False
            missing_files.append(fname)

    if not artifacts_ok:
        logger.info(f"[rebuild] artifacts missing or invalid: building from knowledge_full.md... (missing: {', '.join(missing_files)})")
        if os.path.exists("knowledge_full.md"):
            try:
                build("knowledge_full.md", retries=retries)
            except Exception as e:
                log_and_raise(
                    IndexLoadError,
                    f"Failed to build index: {str(e)}",
                    "check knowledge_full.md file and configuration"
                )
        else:
            log_and_raise(
                IndexLoadError,
                "knowledge_full.md not found",
                "provide a valid knowledge base file to build the index"
            )

    result = load_index()
    if result is None:
        logger.info("[rebuild] artifact validation failed: rebuilding...")
        if os.path.exists("knowledge_full.md"):
            try:
                build("knowledge_full.md", retries=retries)
                result = load_index()
            except Exception as e:
                log_and_raise(
                    IndexLoadError,
                    f"Failed to rebuild index: {str(e)}",
                    "check knowledge_full.md file and configuration"
                )
        else:
            log_and_raise(
                IndexLoadError,
                "knowledge_full.md not found after validation failure",
                "provide a valid knowledge base file to build the index"
            )

    if result is None:
        log_and_raise(
            IndexLoadError,
            "Failed to load artifacts after rebuild",
            "the index may be corrupted; try deleting index files and rebuilding"
        )

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
        log_and_raise(
            TypeError,
            f"load_index() must return dict or tuple, got {type(result)}",
            "contact support - this indicates a system error"
        )

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

    # OPTIMIZATION (Analysis Section 9.1 #3): Load precomputed FAQ cache
    faq_cache = None
    if config.FAQ_CACHE_ENABLED and os.path.exists(config.FAQ_CACHE_PATH):
        try:
            faq_cache = get_precomputed_cache(config.FAQ_CACHE_PATH)
            logger.info(f"FAQ cache loaded: {faq_cache.size()} precomputed answers")
        except Exception as e:
            logger.warning(f"Failed to load FAQ cache: {e}")

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

        # OPTIMIZATION (Analysis Section 9.1 #3): Check FAQ cache first for instant responses
        faq_result = None
        if faq_cache:
            faq_result = faq_cache.get(question, fuzzy=True)

        if faq_result:
            result = {
                "answer": faq_result["answer"],
                "refused": False,
                "confidence": faq_result.get("confidence"),
                "selected_chunks": faq_result.get("packed_chunks", []),
                "metadata": {
                    "used_tokens": 0,
                    "cache_type": "faq_precomputed",
                },
                "routing": {"action": "cache"},
            }
            if debug_enabled:
                print("[DEBUG] FAQ cache hit (precomputed answer)")
        else:
            result = answer_once(
                question,
                chunks,
                vecs_n,
                bm,
                top_k=top_k,
                pack_top=pack_top,
                threshold=threshold,
                use_rerank=use_rerank,
                hnsw=hnsw,
                seed=seed,
                num_ctx=num_ctx,
                num_predict=num_predict,
                retries=retries,
            )

        answer = result["answer"]
        metadata = result.get("metadata", {})
        selected_chunks = result.get("selected_chunks", [])

        if use_json:
            # v4.1: JSON output mode
            used_tokens = metadata.get("used_tokens")
            if used_tokens is None:
                used_tokens = len(selected_chunks)
            output = {
                "answer": answer,
                "citations": selected_chunks,
                "used_tokens": used_tokens,
                "topk": top_k,
                "packed": len(selected_chunks),
                "confidence": result.get("confidence"),
            }
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            print(f"\n{answer}")

        # Show debug info if enabled
        if debug_enabled:
            print(f"\n[DEBUG] Retrieved: {len(selected_chunks)} chunks")
            scores = metadata.get("scores", [])
            if scores:
                print(f"[DEBUG] Scores: {scores[:5]}")

    # Save cache on exit
    query_cache.save()
    logger.info(f"Query cache saved: {len(query_cache.cache)} entries")


def warmup_on_startup():
    """Warm-up embeddings, LLM, and FAISS on startup (reduces first-token latency).

    OPTIMIZATION: Preloads FAISS index to eliminate 50-200ms penalty on first query.
    """
    warmup_enabled = config.WARMUP_ENABLED
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
        client = get_llm_client()
        client.generate_text(
            prompt="Hi",
            model=config.RAG_CHAT_MODEL,
            options={"num_predict": 1, "seed": config.DEFAULT_SEED},
            timeout=(config.CHAT_CONNECT_T, config.CHAT_READ_T),
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


def setup_cli_args():
    """Setup argparse configuration for CLI.

    Returns:
        Parsed arguments namespace
    """
    # Create parent parser for common flags shared across subcommands
    common_flags = argparse.ArgumentParser(add_help=False)
    common_flags.add_argument("--emb-backend", choices=["local", "ollama"], default=config.EMB_BACKEND,
                             help="Embedding backend: local (SentenceTransformer) or ollama (default local)")
    common_flags.add_argument("--ann", choices=["faiss", "none"], default=config.USE_ANN,
                             help="ANN index: faiss (IVFFlat) or none (full-scan, default faiss)")
    common_flags.add_argument("--alpha", type=float, default=config.ALPHA_HYBRID,
                             help="Hybrid scoring blend: alpha*BM25 + (1-alpha)*dense (default 0.5)")

    # Create parent parser for query-related flags shared by 'chat' and 'ask' subcommands
    # FIX (v5.10): Consolidate duplicate flag definitions to single source of truth
    query_flags = argparse.ArgumentParser(add_help=False)
    query_flags.add_argument("--debug", action="store_true", help="Print retrieval diagnostics")
    query_flags.add_argument("--rerank", action="store_true", help="Enable LLM-based reranking")
    query_flags.add_argument("--topk", type=int, default=config.DEFAULT_TOP_K, help="Top-K candidates (default 15)")
    query_flags.add_argument("--pack", type=int, default=config.DEFAULT_PACK_TOP, help="Snippets to pack (default 8)")
    query_flags.add_argument("--threshold", type=float, default=config.DEFAULT_THRESHOLD, help="Cosine threshold (default 0.25)")
    query_flags.add_argument("--seed", type=int, default=config.DEFAULT_SEED, help="Random seed for LLM (default 42)")
    query_flags.add_argument("--num-ctx", type=int, default=config.DEFAULT_NUM_CTX, help=f"LLM context window (default {config.DEFAULT_NUM_CTX})")
    query_flags.add_argument("--num-predict", type=int, default=config.DEFAULT_NUM_PREDICT, help="LLM max generation tokens (default 512)")
    query_flags.add_argument("--retries", type=int, default=config.DEFAULT_RETRIES, help="Retries for transient errors (default 2)")
    query_flags.add_argument("--no-expand", action="store_true",
                             help="Disable query expansion (synonym substitution)")
    query_flags.add_argument("--faiss-multiplier", type=int, default=config.FAISS_CANDIDATE_MULTIPLIER,
                             help="FAISS candidate multiplier: retrieve top_k * N for reranking (default 3)")
    query_flags.add_argument("--json", action="store_true", help="Output answer as JSON with metrics (v4.1)")

    ap = argparse.ArgumentParser(
        prog="clockify_support_cli",
        description="Clockify internal support chatbot (offline, stateless, closed-book)"
    )

    # Global logging and config arguments
    ap.add_argument("--log", default="INFO", choices=["DEBUG", "INFO", "WARN"],
                    help="Logging level (default INFO)")
    ap.add_argument("--no-log", action="store_true",
                    help="Disable query log file writes (privacy mode)")
    ap.add_argument(
        "--ollama-url",
        type=str,
        default=None,
        help=(
            f"Ollama endpoint (default from RAG_OLLAMA_URL env or {config.DEFAULT_RAG_OLLAMA_URL}; "
            f"override with {config.DEFAULT_LOCAL_OLLAMA_URL} for local Ollama)"
        ),
    )
    ap.add_argument("--gen-model", type=str, default=None,
                    help="Generation model name (default from RAG_CHAT_MODEL env or qwen2.5:32b)")
    ap.add_argument("--emb-model", type=str, default=None,
                    help="Embedding model name (default from RAG_EMBED_MODEL env or nomic-embed-text:latest)")
    ap.add_argument("--ctx-budget", type=int, default=None,
                    help="Context token budget (default from CTX_BUDGET env or 12000)")
    ap.add_argument("--query-expansions", type=str, default=None,
                    help="Path to JSON query expansion overrides (default config/query_expansions.json or CLOCKIFY_QUERY_EXPANSIONS env)")
    # Global-only flags
    ap.add_argument("--selftest", action="store_true", help="Run self-tests and exit (v4.1)")
    ap.add_argument("--profile", action="store_true", help="Enable cProfile performance profiling (Rank 29)")

    subparsers = ap.add_subparsers(dest="cmd")

    # Build subparser with common flags
    b = subparsers.add_parser("build", help="Build knowledge base", parents=[common_flags])
    b.add_argument("md_path", help="Path to knowledge_full.md")
    b.add_argument("--retries", type=int, default=config.DEFAULT_RETRIES, help="Retries for transient errors (default 2)")

    # Chat subparser with common flags and query flags
    c = subparsers.add_parser("chat", help="Start REPL", parents=[common_flags, query_flags])
    c.add_argument("--det-check", action="store_true", help="Determinism check: ask same Q twice, compare hashes")

    # Ask subparser with common flags and query flags
    a = subparsers.add_parser("ask", help="Answer a single question and exit", parents=[common_flags, query_flags])
    a.add_argument("question", help="Question to answer")

    return ap.parse_args()


def configure_logging_and_config(args):
    """Configure logging, query expansion, and global config from CLI args.

    Args:
        args: Parsed argparse namespace

    Returns:
        query_log_disabled: Boolean flag for query logging
    """
    # Setup logging after CLI arg parsing
    level = getattr(logging, args.log if hasattr(args, "log") else "INFO")
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    # Configure query expansion dictionary overrides
    if getattr(args, "query_expansions", None):
        set_query_expansion_path(args.query_expansions)
    elif config.CLOCKIFY_QUERY_EXPANSIONS:
        set_query_expansion_path(config.CLOCKIFY_QUERY_EXPANSIONS)
    else:
        set_query_expansion_path(None)

    try:
        load_query_expansion_dict(force_reload=True, suppress_errors=False)
    except ValueError as exc:
        logger.error("CONFIG ERROR: %s", exc)
        sys.exit(1)

    query_log_disabled = getattr(args, "no_log", False)

    # Update globals from CLI args
    config.EMB_BACKEND = getattr(args, 'emb_backend', config.EMB_BACKEND)
    config.USE_ANN = getattr(args, 'ann', config.USE_ANN)
    config.ALPHA_HYBRID = getattr(args, 'alpha', config.ALPHA_HYBRID)

    # Update FAISS multiplier if provided in subcommand args
    if hasattr(args, "faiss_multiplier"):
        config.FAISS_CANDIDATE_MULTIPLIER = args.faiss_multiplier

    # Validate and set config from CLI args
    try:
        validate_and_set_config(
            ollama_url=args.ollama_url,
            gen_model=args.gen_model,
            emb_model=args.emb_model,
            ctx_budget=args.ctx_budget
        )
        validate_chunk_config()
        check_pytorch_mps()
    except ValueError as e:
        logger.error(f"CONFIG ERROR: {e}")
        sys.exit(1)

    return query_log_disabled


def handle_build_command(args):
    """Handle build command."""
    build(args.md_path, retries=getattr(args, "retries", 0))


def handle_ask_command(args):
    """Handle ask command."""
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
    result = answer_once(
        args.question,
        chunks,
        vecs_n,
        bm,
        top_k=args.topk,
        pack_top=args.pack,
        threshold=args.threshold,
        use_rerank=args.rerank,
        hnsw=hnsw,
        seed=args.seed,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        retries=getattr(args, "retries", 0)
    )
    metadata = result.get("metadata", {})
    timing = result.get("timing")
    routing = result.get("routing")
    selected_chunks = result.get("selected_chunks", [])
    ans = result["answer"]
    refused = result.get("refused", False)
    if getattr(args, "json", False):
        used_tokens = metadata.get("used_tokens")
        if used_tokens is None:
            used_tokens = len(selected_chunks)
        output = answer_to_json(
            ans,
            selected_chunks,
            used_tokens,
            args.topk,
            args.pack,
            result.get("confidence"),
            metadata=metadata,
            routing=routing,
            timing=timing,
            refused=refused,
        )
        output["question"] = args.question
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(ans)


def handle_chat_command(args):
    """Handle chat command including determinism check."""
    # Determinism check
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
            # Determinism check using Ollama
            try:
                seed = 42
                np.random.seed(seed)
                prompt = "What is Clockify?"
                client = get_llm_client()
                messages: List[ChatMessage] = [
                    {"role": "system", "content": "Determinism check"},
                    {"role": "user", "content": prompt},
                ]
                options: ChatCompletionOptions = {
                    "seed": seed,
                    "num_ctx": config.DEFAULT_NUM_CTX,
                    "num_predict": config.DEFAULT_NUM_PREDICT,
                }

                resp1 = client.chat_completion(
                    messages=messages,
                    model=config.RAG_CHAT_MODEL,
                    options=dict(options),
                    stream=False,
                    timeout=(config.CHAT_CONNECT_T, config.CHAT_READ_T),
                    retries=retries or config.DEFAULT_RETRIES,
                )
                ans1 = resp1.get("message", {}).get("content", "")

                np.random.seed(seed)
                resp2 = client.chat_completion(
                    messages=messages,
                    model=config.RAG_CHAT_MODEL,
                    options=dict(options),
                    stream=False,
                    timeout=(config.CHAT_CONNECT_T, config.CHAT_READ_T),
                    retries=retries or config.DEFAULT_RETRIES,
                )
                ans2 = resp2.get("message", {}).get("content", "")

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
        use_json=getattr(args, "json", False)
    )
