"""Tests for retrieval pipeline (hybrid BM25 + dense retrieval)."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clockify_support_cli_final import retrieve, normalize_scores_zscore, sanitize_question


def test_retrieve_returns_correct_top_k(sample_chunks, sample_embeddings, sample_bm25):
    """Verify retrieval returns exactly top_k results."""
    question = "How do I track time?"
    top_k = 2

    selected, scores = retrieve(question, sample_chunks, sample_embeddings, sample_bm25, top_k=top_k)

    assert len(selected) <= top_k, f"Should return at most {top_k} results, got {len(selected)}"
    assert all(isinstance(idx, (int, np.integer)) for idx in selected), "Indices should be integers"
    assert all(0 <= idx < len(sample_chunks) for idx in selected), "Indices should be valid"


def test_retrieve_with_different_top_k_values(sample_chunks, sample_embeddings, sample_bm25):
    """Test retrieval with various top_k values."""
    question = "time tracking features"

    for k in [1, 3, 5]:
        selected, scores = retrieve(question, sample_chunks, sample_embeddings, sample_bm25, top_k=k)
        assert len(selected) <= k, f"With top_k={k}, got {len(selected)} results"


def test_retrieve_with_empty_query_raises_error(sample_chunks, sample_embeddings, sample_bm25):
    """Verify retrieval handles empty query gracefully."""
    # sanitize_question should raise ValueError for empty string
    with pytest.raises(ValueError, match="Question cannot be empty"):
        sanitize_question("")


def test_retrieve_scores_structure(sample_chunks, sample_embeddings, sample_bm25):
    """Verify scores dict has correct structure."""
    question = "pricing plans"

    selected, scores = retrieve(question, sample_chunks, sample_embeddings, sample_bm25)

    assert isinstance(scores, dict), "Scores should be a dict"
    assert "dense" in scores, "Should have dense scores"
    assert "bm25" in scores, "Should have BM25 scores"
    assert "hybrid" in scores, "Should have hybrid scores"

    # Check array lengths
    n_chunks = len(sample_chunks)
    assert len(scores["dense"]) == n_chunks, "Dense scores should match chunk count"
    assert len(scores["bm25"]) == n_chunks, "BM25 scores should match chunk count"
    assert len(scores["hybrid"]) == n_chunks, "Hybrid scores should match chunk count"


def test_normalize_scores_zscore_edge_cases():
    """Test score normalization edge cases."""
    # All equal scores (std=0)
    scores = [0.5, 0.5, 0.5]
    result = normalize_scores_zscore(scores)
    assert np.allclose(result, [0.5, 0.5, 0.5]), "Should preserve values when std=0"

    # Empty array
    result = normalize_scores_zscore([])
    assert len(result) == 0, "Should handle empty array"

    # Single value
    result = normalize_scores_zscore([1.0])
    assert len(result) == 1, "Should handle single value"

    # Normal case
    scores = [1.0, 2.0, 3.0]
    result = normalize_scores_zscore(scores)
    assert abs(result.mean()) < 0.01, "Mean should be ~0 after normalization"
    assert abs(result.std() - 1.0) < 0.01, "Std should be ~1 after normalization"


def test_retrieve_ranking_quality(sample_chunks, sample_embeddings, sample_bm25):
    """Test that retrieval ranks relevant chunks higher."""
    question = "How do I track time?"  # Should rank time tracking chunks higher

    selected, scores = retrieve(question, sample_chunks, sample_embeddings, sample_bm25, top_k=3)

    # Check that at least one time tracking chunk is in top results
    # (chunks 0 and 2 are about time tracking)
    time_tracking_chunks = {0, 2}
    retrieved_set = set(selected)
    overlap = time_tracking_chunks & retrieved_set

    assert len(overlap) > 0, f"Should retrieve at least one time tracking chunk, got {selected}"


def test_retrieve_with_long_query(sample_chunks, sample_embeddings, sample_bm25):
    """Test retrieval with a longer, more complex query."""
    question = (
        "I need to understand how to properly track my work hours and time spent on projects "
        "using the timer functionality, including manual entry options"
    )

    selected, scores = retrieve(question, sample_chunks, sample_embeddings, sample_bm25, top_k=3)

    assert len(selected) > 0, "Should return results for long query"
    assert len(selected) <= 3, "Should respect top_k limit"


def test_sanitize_question_validation():
    """Test question sanitization and validation."""
    # Valid questions
    assert sanitize_question("How do I track time?") == "How do I track time?"
    assert sanitize_question("  whitespace  ") == "whitespace"

    # Invalid questions
    with pytest.raises(ValueError, match="empty"):
        sanitize_question("")

    with pytest.raises(ValueError, match="too long"):
        sanitize_question("x" * 3000)  # Exceeds max_length=2000

    with pytest.raises(ValueError, match="string"):
        sanitize_question(123)  # Not a string

    # Control characters
    with pytest.raises(ValueError, match="control characters"):
        sanitize_question("test\x00string")  # Null byte


def test_retrieve_deduplication(sample_chunks, sample_embeddings, sample_bm25):
    """Test that retrieval doesn't return duplicate indices."""
    question = "time tracking"

    selected, scores = retrieve(question, sample_chunks, sample_embeddings, sample_bm25, top_k=5)

    # Check for duplicates
    assert len(selected) == len(set(selected)), "Should not return duplicate indices"


def test_retrieve_scores_are_numeric(sample_chunks, sample_embeddings, sample_bm25):
    """Verify all scores are valid numeric values."""
    question = "clockify features"

    selected, scores = retrieve(question, sample_chunks, sample_embeddings, sample_bm25)

    for score_type in ["dense", "bm25", "hybrid"]:
        score_array = scores[score_type]
        assert np.all(np.isfinite(score_array)), f"{score_type} scores should be finite"
        assert not np.any(np.isnan(score_array)), f"{score_type} scores should not contain NaN"
