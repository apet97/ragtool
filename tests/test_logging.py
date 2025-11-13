import json

import clockify_rag.caching as caching
from clockify_rag.caching import log_query
import clockify_rag.config


def test_log_query_records_non_zero_scores(tmp_path, monkeypatch):
    """Ensure log entries include non-zero dense/BM25 scores."""
    log_path = tmp_path / "rag_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    caching._reset_query_logger_for_tests()
    # Monkeypatch at the config module level where log_query() imports from
    monkeypatch.setattr(clockify_rag.config, "QUERY_LOG_FILE", str(log_path))

    monkeypatch.setattr(clockify_rag.config, "QUERY_LOG_MAX_BYTES", 1024)
    monkeypatch.setattr(clockify_rag.config, "QUERY_LOG_BACKUP_COUNT", 1)

    retrieved_chunks = [
        {"id": "chunk-1", "dense": 0.42, "bm25": 0.15, "hybrid": 0.30},
        {"id": "chunk-2", "dense": 0.10, "bm25": 0.05, "hybrid": 0.08},
    ]

    log_query(
        query="What is Clockify?",
        answer="Clockify is a time tracker.",
        retrieved_chunks=retrieved_chunks,
        latency_ms=120,
        refused=False,
        metadata={"test": True},
    )

    log_lines = log_path.read_text().strip().splitlines()
    assert log_lines, "Log file should contain at least one entry"

    entry = json.loads(log_lines[-1])
    assert entry["retrieved_chunks"][0]["id"] == "chunk-1"
    assert entry["retrieved_chunks"][0]["dense"] > 0
    assert entry["retrieved_chunks"][0]["bm25"] > 0
    assert entry["chunk_scores"]["dense"][0] > 0
    assert entry["chunk_scores"]["bm25"][0] > 0


def test_query_log_rotates_and_respects_retention(tmp_path, monkeypatch):
    """Query log writes should rotate and keep only configured backups."""

    log_path = tmp_path / "queries.jsonl"

    caching._reset_query_logger_for_tests()
    monkeypatch.setattr(clockify_rag.config, "QUERY_LOG_FILE", str(log_path))
    monkeypatch.setattr(clockify_rag.config, "QUERY_LOG_MAX_BYTES", 200)
    monkeypatch.setattr(clockify_rag.config, "QUERY_LOG_BACKUP_COUNT", 2)

    retrieved_chunks = [
        {"id": "chunk-1", "dense": 0.9, "bm25": 0.5, "hybrid": 0.7}
    ]

    large_query = "?" * 400

    for idx in range(15):
        log_query(
            query=f"{large_query}-{idx}",
            answer="Clockify is a time tracker.",
            retrieved_chunks=retrieved_chunks,
            latency_ms=42.0,
            refused=False,
            metadata={"test": True, "iteration": idx},
        )

    primary = log_path
    backups = sorted(tmp_path.glob("queries.jsonl.*"))

    assert primary.exists(), "Current log file should exist after rotation"
    assert backups, "Rotation should create at least one backup file"
    assert len(backups) <= 2, "Should not retain more backups than configured"

    # Most recent backup is .1; ensure it contains data
    assert backups[0].stat().st_size > 0
