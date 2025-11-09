"""Shared pytest fixtures for RAG system tests."""

import pytest
import numpy as np
import json
import tempfile
import os
import sys

# Import from refactored package modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_configure(config):
    """Validate test environment has required dependencies.

    This hook runs before any tests and ensures all required dependencies
    are installed. It exits with an error if critical dependencies are missing,
    and warns about optional dependencies.
    """
    missing = []

    # Check required dependencies
    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        import requests
    except ImportError:
        missing.append("requests")

    if missing:
        pytest.exit(
            f"\nERROR: Missing required test dependencies: {', '.join(missing)}\n"
            f"Install with: pip install -e '.[dev]'\n",
            returncode=1
        )

    # Check optional dependencies (just warn)
    optional_missing = []
    try:
        import faiss
    except ImportError:
        optional_missing.append("faiss-cpu")

    try:
        import torch
    except ImportError:
        optional_missing.append("torch")

    if optional_missing and config.option.verbose >= 0:
        print(f"\nWARNING: Optional dependencies not installed: {', '.join(optional_missing)}")
        print("Some tests may be skipped. Install with: pip install -e '.[dev]'\n")


from clockify_rag.indexing import build_bm25
from clockify_rag.config import DEFAULT_TOP_K


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        {
            "id": "1",
            "title": "Time Tracking",
            "section": "Basics",
            "text": "Track time by clicking the timer button in the top right corner",
            "url": "https://clockify.me/help/time-tracking"
        },
        {
            "id": "2",
            "title": "Pricing",
            "section": "Plans",
            "text": "Free plan includes unlimited users and unlimited time tracking",
            "url": "https://clockify.me/pricing"
        },
        {
            "id": "3",
            "title": "Time Tracking",
            "section": "Advanced",
            "text": "Manual time entries can be added by clicking the plus button",
            "url": "https://clockify.me/help/time-tracking-advanced"
        },
        {
            "id": "4",
            "title": "Reports",
            "section": "Overview",
            "text": "Generate detailed reports of your time tracking data",
            "url": "https://clockify.me/help/reports"
        },
        {
            "id": "5",
            "title": "Integrations",
            "section": "Setup",
            "text": "Integrate Clockify with your favorite tools like Jira and Slack",
            "url": "https://clockify.me/help/integrations"
        }
    ]


@pytest.fixture
def sample_embeddings(sample_chunks):
    """Sample embeddings for testing (384-dim for local, 768-dim for Ollama).

    Creates simple embeddings with some semantic similarity between related chunks.
    """
    from clockify_rag.config import EMB_DIM
    n = len(sample_chunks)

    # Create base embeddings
    vecs = np.random.randn(n, EMB_DIM).astype("float32")

    # Add some similarity between time tracking chunks (0 and 2)
    vecs[2] = vecs[0] + np.random.randn(EMB_DIM).astype("float32") * 0.1

    # Normalize all vectors
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


@pytest.fixture
def sample_bm25(sample_chunks):
    """Sample BM25 index for testing."""
    return build_bm25(sample_chunks)


@pytest.fixture
def temp_index_dir(sample_chunks, sample_embeddings, sample_bm25):
    """Temporary directory with full index artifacts.

    Creates a temporary directory with:
    - chunks.jsonl
    - vecs_n.npy
    - bm25.json
    - index.meta.json
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write chunks
        chunks_path = os.path.join(tmpdir, "chunks.jsonl")
        with open(chunks_path, "w") as f:
            for c in sample_chunks:
                f.write(json.dumps(c) + "\n")

        # Write embeddings
        vecs_path = os.path.join(tmpdir, "vecs_n.npy")
        np.save(vecs_path, sample_embeddings)

        # Write BM25
        bm25_path = os.path.join(tmpdir, "bm25.json")
        with open(bm25_path, "w") as f:
            json.dump(sample_bm25, f)

        # Write metadata
        meta = {
            "chunks": len(sample_chunks),
            "emb_rows": len(sample_embeddings),
            "bm25_docs": len(sample_bm25["doc_lens"]),
            "kb_sha256": "test_hash",
            "built_at": "2025-11-06T00:00:00Z"
        }
        meta_path = os.path.join(tmpdir, "index.meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        yield tmpdir


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "How do I track time?"


@pytest.fixture
def sample_query_embedding():
    """Sample query embedding for testing."""
    from clockify_rag.config import EMB_DIM
    qv = np.random.randn(EMB_DIM).astype("float32")
    qv = qv / np.linalg.norm(qv)
    return qv


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response for testing."""
    return {
        "embedding": np.random.randn(768).tolist(),  # 768-dim for Ollama
        "model": "nomic-embed-text"
    }


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "message": {
            "content": "Track time by clicking the timer button in the top right corner."
        },
        "model": "qwen2.5:32b",
        "done": True
    }
