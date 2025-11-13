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

# Standard library imports
import atexit
import hashlib
import json
import logging
import os
import sys
import tempfile
import threading
import time
from contextlib import ExitStack, contextmanager
from typing import Any, Callable, Dict, List

# Third-party imports
import numpy as np

# Package imports
import clockify_rag.config as config
from clockify_rag.caching import get_query_cache, get_rate_limiter, QueryCache, RateLimiter, log_query
from clockify_rag.utils import _release_lock_if_owner, _log_config_summary
from clockify_rag.exceptions import EmbeddingError, LLMError, IndexLoadError, BuildError
from clockify_rag.cli import (
    setup_cli_args,
    configure_logging_and_config,
    handle_build_command,
    handle_ask_command,
    handle_chat_command,
    chat_repl,
    warmup_on_startup
)

# Re-export config constants and functions for backward compatibility with tests
from clockify_rag.config import (
    LOG_QUERY_INCLUDE_CHUNKS,
    QUERY_LOG_FILE,
)

# Re-export functions used by tests
from clockify_rag.answer import answer_once
import clockify_rag.retrieval as retrieval_module
from clockify_rag.retrieval import retrieve, coverage_ok
from clockify_rag.indexing import build, build_bm25, load_index
from clockify_rag.answer import (
    apply_mmr_diversification,
    apply_reranking,
    pack_snippets,
    generate_llm_answer,
)
from clockify_rag.utils import inject_policy_preamble

# ====== MODULE GLOBALS ======
logger = logging.getLogger(__name__)
QUERY_LOG_DISABLED = False  # Can be set to True via --no-log flag
atexit.register(_release_lock_if_owner)

# Global instances
RATE_LIMITER = get_rate_limiter()
QUERY_CACHE = get_query_cache()


# ====== SELF-TEST SUPPORT ======


class _SyntheticIndexFixture:
    """Create a minimal synthetic index on disk for self-tests."""

    def __init__(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.base_path = self._tempdir.name
        self.embedding_dim = config.EMB_DIM_LOCAL if config.EMB_BACKEND == "local" else config.EMB_DIM_OLLAMA
        self.chunks: List[Dict[str, Any]] = [
            {
                "id": "chunk-time-tracking",
                "title": "Time Tracking",
                "section": "Basics",
                "text": "Clockify lets teams track time entries across projects with tags and billable rates.",
            },
            {
                "id": "chunk-reporting",
                "title": "Reporting",
                "section": "Overview",
                "text": "Detailed reports highlight billable hours, budgets, and project performance in Clockify.",
            },
        ]
        self.chunk_ids = [chunk["id"] for chunk in self.chunks]
        self._artifacts_cache: Dict[str, Any] | None = None
        self._write_artifacts()

    def _write_artifacts(self) -> None:
        def _write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
            with open(path, "w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record))
                    handle.write("\n")

        chunks_path = os.path.join(self.base_path, config.FILES["chunks"])
        _write_jsonl(chunks_path, self.chunks)

        meta_records = [
            {"id": chunk["id"], "title": chunk["title"], "section": chunk["section"]}
            for chunk in self.chunks
        ]
        meta_path = os.path.join(self.base_path, config.FILES["meta"])
        _write_jsonl(meta_path, meta_records)

        vecs = np.zeros((len(self.chunks), self.embedding_dim), dtype="float32")
        if len(self.chunks) >= 1:
            vecs[0, 0] = 1.0
        if len(self.chunks) >= 2 and self.embedding_dim > 1:
            vecs[1, 1] = 1.0
        emb_path = os.path.join(self.base_path, config.FILES["emb"])
        np.save(emb_path, vecs)

        bm25_data = build_bm25(self.chunks)
        bm25_path = os.path.join(self.base_path, config.FILES["bm25"])
        with open(bm25_path, "w", encoding="utf-8") as handle:
            json.dump(bm25_data, handle)

        index_meta = {
            "source": "selftest",
            "emb_backend": config.EMB_BACKEND,
            "emb_model": config.EMB_MODEL if config.EMB_BACKEND != "local" else "all-MiniLM-L6-v2",
            "emb_dim": self.embedding_dim,
            "chunks": len(self.chunks),
        }
        index_meta_path = os.path.join(self.base_path, config.FILES["index_meta"])
        with open(index_meta_path, "w", encoding="utf-8") as handle:
            json.dump(index_meta, handle)

    @contextmanager
    def _cwd(self):
        prev_cwd = os.getcwd()
        try:
            os.chdir(self.base_path)
            yield
        finally:
            os.chdir(prev_cwd)

    def load_index(self) -> Dict[str, Any] | None:
        with self._cwd():
            return load_index()

    def artifacts(self) -> Dict[str, Any]:
        if self._artifacts_cache is None:
            loaded = self.load_index()
            if loaded is None:
                raise IndexLoadError("Synthetic index failed to load")
            self._artifacts_cache = loaded
        return self._artifacts_cache

    def cleanup(self) -> None:
        self._tempdir.cleanup()


@contextmanager
def _synthetic_fixture() -> Any:
    fixture = _SyntheticIndexFixture()
    try:
        yield fixture
    finally:
        fixture.cleanup()


@contextmanager
def _patch_attr(obj: Any, attr: str, value: Any):
    original = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, original)


def _fake_query_embed_factory(dim: int) -> Callable[[str, int], np.ndarray]:
    def _fake_embed(question: str, retries: int = 0) -> np.ndarray:
        _ = retries
        vec = np.zeros(dim, dtype="float32")
        lowered = question.lower()
        hits = 0
        if "time" in lowered or "track" in lowered:
            vec[0] = 1.0
            hits += 1
        if dim > 1 and ("report" in lowered or "reporting" in lowered):
            vec[1] = 1.0
            hits += 1
        if hits == 0:
            vec[0] = 1.0
            hits = 1
        if hits > 1:
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
        return vec

    return _fake_embed


def _selftest_index_loads(fixture: _SyntheticIndexFixture) -> bool:
    artifacts = fixture.load_index()
    if artifacts is None:
        return False
    chunks = artifacts.get("chunks", [])
    vecs = artifacts.get("vecs_n")
    bm = artifacts.get("bm")
    return (
        isinstance(chunks, list)
        and len(chunks) == len(fixture.chunks)
        and isinstance(vecs, np.ndarray)
        and vecs.shape[1] == fixture.embedding_dim
        and isinstance(bm, dict)
    )


def _selftest_retrieval_smoke(fixture: _SyntheticIndexFixture) -> bool:
    artifacts = fixture.artifacts()
    indices, _scores = retrieve(
        "How do I track time entries?",
        artifacts["chunks"],
        artifacts["vecs_n"],
        artifacts["bm"],
        top_k=2,
    )
    if not indices:
        return False
    top_chunk_id = artifacts["chunks"][indices[0]]["id"]
    return top_chunk_id == fixture.chunk_ids[0]


def _selftest_coverage_guard(fixture: _SyntheticIndexFixture) -> bool:
    artifacts = fixture.artifacts()
    indices, scores = retrieve(
        "Need time tracking reporting guidance",
        artifacts["chunks"],
        artifacts["vecs_n"],
        artifacts["bm"],
        top_k=2,
    )
    if len(indices) < 2:
        return False
    return coverage_ok(indices[:2], scores["dense"], threshold=0.2)


def run_selftest() -> bool:
    """Run lightweight diagnostics covering index loading and retrieval flows."""

    logger.info("[selftest] Starting diagnostics")
    results: List[bool] = []

    try:
        with _synthetic_fixture() as fixture, ExitStack() as stack:
            stack.enter_context(_patch_attr(config, "USE_ANN", "none"))
            fake_embed = _fake_query_embed_factory(fixture.embedding_dim)
            stack.enter_context(_patch_attr(retrieval_module, "_embedding_embed_query", fake_embed))

            tests: List[tuple[str, Callable[[], bool]]] = [
                ("Synthetic index loads", lambda: _selftest_index_loads(fixture)),
                ("Hybrid retrieval smoke", lambda: _selftest_retrieval_smoke(fixture)),
                ("Coverage guard", lambda: _selftest_coverage_guard(fixture)),
            ]

            for name, test_fn in tests:
                try:
                    ok = bool(test_fn())
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.exception("[selftest] %s: FAIL (%s)", name, exc)
                    ok = False
                else:
                    logger.info("[selftest] %s: %s", name, "PASS" if ok else "FAIL")
                results.append(ok)

    except Exception as exc:  # pragma: no cover - catastrophic failure
        logger.exception("[selftest] Fatal error: %s", exc)
        return False

    passed = sum(1 for r in results if r)
    total = len(results)
    logger.info("[selftest] %d/%d tests passed", passed, total)
    return bool(results) and all(results)


# ====== MAIN ENTRY POINT ======

def main():
    """Main entry point - delegates to CLI module for all functionality."""
    global QUERY_LOG_DISABLED

    # Parse command line arguments
    args = setup_cli_args()

    # Lightweight selftest path: skip full config wiring when only --selftest is provided
    if getattr(args, "selftest", False) and args.cmd is None:
        log_level = getattr(logging, getattr(args, "log", "INFO").upper(), logging.INFO)
        logging.basicConfig(level=log_level)
        QUERY_LOG_DISABLED = True
        success = run_selftest()
        sys.exit(0 if success else 1)

    # Configure logging and global config
    QUERY_LOG_DISABLED = configure_logging_and_config(args)

    # Handle selftest if requested
    if getattr(args, "selftest", False):
        success = run_selftest()
        sys.exit(0 if success else 1)

    # Auto-start REPL if no command given
    if args.cmd is None:
        try:
            chat_repl()
        except IndexLoadError as exc:
            logger.error("Failed to load index: %s", exc)
            sys.exit(getattr(exc, "exit_code", 1))
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
