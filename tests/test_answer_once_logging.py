import numpy as np
import pytest

import clockify_support_cli_final as cli


def test_answer_once_logs_retrieved_chunks_with_cache(monkeypatch):
    cli.QUERY_CACHE.clear()

    # Enable chunk logging for this test by monkeypatching the module constant
    monkeypatch.setattr(cli, "LOG_QUERY_INCLUDE_CHUNKS", True)

    # Ensure rate limiter allows requests during the test
    monkeypatch.setattr(cli.RATE_LIMITER, "allow_request", lambda: True)
    monkeypatch.setattr(cli.RATE_LIMITER, "wait_time", lambda: 0)

    chunks = [
        {
            "id": "chunk-1",
            "title": "Test",
            "section": "Section",
            "url": "http://example.com",
            "text": "Example chunk text",
        }
    ]
    vecs_n = np.zeros((1, 3), dtype=np.float32)

    def fake_retrieve(question, chunks_arg, vecs_arg, bm, top_k, hnsw, retries):
        return [0], {
            "dense": np.array([0.9], dtype=np.float32),
            "bm25": np.array([0.1], dtype=np.float32),
            "hybrid": np.array([0.5], dtype=np.float32),
        }

    monkeypatch.setattr(cli, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        cli,
        "apply_mmr_diversification",
        lambda selected, scores, vecs_arg, pack_top: selected,
    )
    monkeypatch.setattr(
        cli,
        "apply_reranking",
        lambda question, chunks_arg, mmr_selected, scores, use_rerank, seed, num_ctx, num_predict, retries: (mmr_selected, {}, False, "", 0.0),
    )
    monkeypatch.setattr(cli, "coverage_ok", lambda selected, dense_scores, threshold: True)

    def fake_pack_snippets(chunks_arg, selected, pack_top, budget_tokens, num_ctx):
        return "context", [chunks_arg[i]["id"] for i in selected], 12

    monkeypatch.setattr(cli, "pack_snippets", fake_pack_snippets)
    monkeypatch.setattr(cli, "inject_policy_preamble", lambda block, question: block)
    monkeypatch.setattr(cli, "generate_llm_answer", lambda *args, **kwargs: ("answer", 0.01, 88))

    logged_calls = []

    def fake_log_query(query, answer, retrieved_chunks, latency_ms, refused=False, metadata=None):
        logged_calls.append(
            {
                "query": query,
                "answer": answer,
                "retrieved_chunks": retrieved_chunks,
                "latency_ms": latency_ms,
                "refused": refused,
                "metadata": metadata,
            }
        )

    monkeypatch.setattr(cli, "log_query", fake_log_query)

    answer, metadata = cli.answer_once(
        "What is the chunk?",
        chunks,
        vecs_n,
        bm=None,
        top_k=1,
        pack_top=1,
        threshold=0.1,
        use_rerank=False,
        debug=False,
        hnsw=None,
    )

    assert answer == "answer"
    assert metadata["cached"] is False
    assert metadata["cache_hit"] is False
    assert len(logged_calls) == 1

    retrieved_chunks = logged_calls[0]["retrieved_chunks"]
    assert len(retrieved_chunks) == 1
    chunk_entry = retrieved_chunks[0]
    assert chunk_entry["id"] == "chunk-1"
    assert chunk_entry["chunk"] == chunks[0]
    # In success path, code uses "dense" not "score"
    assert chunk_entry["dense"] == pytest.approx(0.9)

    # Second call should hit the cache and avoid invoking log_query again
    cached_answer, cached_metadata = cli.answer_once(
        "What is the chunk?",
        chunks,
        vecs_n,
        bm=None,
        top_k=1,
        pack_top=1,
        threshold=0.1,
        use_rerank=False,
        debug=False,
        hnsw=None,
    )

    assert cached_answer == "answer"
    assert cached_metadata["cached"] is True
    assert cached_metadata["cache_hit"] is True
    assert len(logged_calls) == 1
