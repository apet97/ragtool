"""Helpers for structured query logging across CLI and API surfaces."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .caching import log_query


def _normalize_chunk_dict(chunk: Mapping[str, Any], rank: int) -> Dict[str, Any]:
    """Return a sanitized chunk dict for logging."""

    normalized = {
        "id": chunk.get("id") or chunk.get("chunk_id"),
        "title": chunk.get("title"),
        "section": chunk.get("section"),
        "rank": rank,
    }

    # Carry optional identifiers for easier debugging
    if "url" in chunk:
        normalized["url"] = chunk["url"]
    if "score" in chunk:
        normalized["score"] = chunk["score"]
    if "dense" in chunk:
        normalized["dense"] = chunk["dense"]
    if "bm25" in chunk:
        normalized["bm25"] = chunk["bm25"]
    if "hybrid" in chunk:
        normalized["hybrid"] = chunk["hybrid"]

    return normalized


def build_chunk_log_entries(
    chunks: Sequence[Mapping[str, Any]] | None,
    selected_chunks: Sequence[Any] | None,
) -> List[Dict[str, Any]]:
    """Convert selected chunk references into structured log entries."""

    if not selected_chunks:
        return []

    log_entries: List[Dict[str, Any]] = []
    total_chunks = len(chunks) if chunks is not None else 0

    for rank, chunk_ref in enumerate(selected_chunks):
        if isinstance(chunk_ref, Mapping):
            entry = _normalize_chunk_dict(chunk_ref, rank)
            if entry.get("id") is None and "chunk_id" in chunk_ref:
                entry["id"] = chunk_ref["chunk_id"]
            log_entries.append(entry)
            continue

        try:
            idx = int(chunk_ref)
        except (TypeError, ValueError):
            log_entries.append({"id": chunk_ref, "rank": rank})
            continue

        if 0 <= idx < total_chunks and chunks is not None:
            chunk = chunks[idx]
            entry = {
                "id": chunk.get("id", idx),
                "title": chunk.get("title"),
                "section": chunk.get("section"),
                "rank": rank,
            }
            if "url" in chunk:
                entry["url"] = chunk["url"]
            log_entries.append(entry)
        else:
            log_entries.append({"id": idx, "rank": rank})

    return log_entries


def log_query_event(
    question: str,
    result: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]] | None,
    latency_ms: Optional[float],
    *,
    channel: Optional[str] = None,
    disabled: bool = False,
) -> None:
    """Write a structured query log entry when logging is enabled."""

    if disabled:
        return

    if not question or not isinstance(result, Mapping):
        return

    retrieved_chunks = build_chunk_log_entries(chunks, result.get("selected_chunks"))
    timing = result.get("timing") or {}
    computed_latency = latency_ms if latency_ms is not None else timing.get("total_ms")

    metadata: Dict[str, Any] = {
        "confidence": result.get("confidence"),
        "routing": result.get("routing"),
        "result_metadata": result.get("metadata") or {},
        "timing": timing,
    }
    if channel:
        metadata["channel"] = channel

    log_query(
        question,
        result.get("answer", ""),
        retrieved_chunks,
        float(computed_latency) if computed_latency is not None else 0.0,
        refused=bool(result.get("refused")),
        metadata=metadata,
    )
