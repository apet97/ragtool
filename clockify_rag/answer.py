"""Answer generation module for Clockify RAG system.

Priority #6: Split monolithic CLI (ROI 7/10)

This module contains the complete answer generation pipeline:
- MMR diversification
- Optional LLM reranking
- Answer generation with confidence scoring
- Citation validation
- Complete answer_once workflow
"""

import json
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Mapping, Sequence

import numpy as np

from .config import (
    DEFAULT_TOP_K,
    DEFAULT_PACK_TOP,
    DEFAULT_THRESHOLD,
    DEFAULT_SEED,
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_PREDICT,
    DEFAULT_RETRIES,
    MMR_LAMBDA,
    REFUSAL_STR,
    STRICT_CITATIONS,
)
from .retrieval import (
    retrieve,
    rerank_with_llm,
    pack_snippets,
    coverage_ok,
    ask_llm,
)
from .exceptions import LLMError
from .confidence_routing import get_routing_action
from .metrics import increment_counter, observe_histogram, MetricNames

logger = logging.getLogger(__name__)


def _extract_chunk_identifier(chunk: Mapping[str, Any], fallback: Any) -> Any:
    """Return a stable chunk identifier for telemetry and caching."""

    for key in ("id", "chunk_id"):
        if key in chunk and chunk[key] is not None:
            return chunk[key]
    return fallback


def _resolve_chunk_ids(chunks: Sequence[Mapping[str, Any]], indices: Sequence[Any]) -> List[Any]:
    """Map positional indices to chunk identifiers."""

    resolved: List[Any] = []
    total = len(chunks)

    for idx in indices:
        if isinstance(idx, Mapping):
            resolved.append(_extract_chunk_identifier(idx, idx))
            continue

        try:
            pos = int(idx)
        except (TypeError, ValueError):
            resolved.append(idx)
            continue

        if 0 <= pos < total:
            resolved.append(_extract_chunk_identifier(chunks[pos], pos))
        else:
            resolved.append(pos)

    return resolved


def apply_mmr_diversification(
    selected: List[int],
    scores: Dict[str, Any],
    vecs_n: np.ndarray,
    pack_top: int
) -> List[int]:
    """Apply Maximal Marginal Relevance diversification to selected chunks.

    Args:
        selected: List of selected chunk indices
        scores: Dict with "dense" scores
        vecs_n: Normalized embedding vectors
        pack_top: Maximum number of chunks to select

    Returns:
        List of diversified chunk indices
    """
    mmr_selected = []
    cand = list(selected)

    # Always include the top dense score first for better recall
    if cand:
        top_dense_idx = max(cand, key=lambda j: scores["dense"][j])
        mmr_selected.append(top_dense_idx)
        cand.remove(top_dense_idx)

    # Then diversify the rest using vectorized MMR
    if cand and len(mmr_selected) < pack_top:
        # Convert to numpy arrays for vectorized operations
        cand_array = np.array(cand, dtype=np.int32)
        relevance_scores = np.array([scores["dense"][j] for j in cand], dtype=np.float32)

        # Get embedding vectors for candidates
        cand_vecs = vecs_n[cand_array]  # [num_candidates, emb_dim]

        # Iteratively select using MMR
        remaining_mask = np.ones(len(cand_array), dtype=bool)

        while np.any(remaining_mask) and len(mmr_selected) < pack_top:
            # Compute MMR scores for remaining candidates
            mmr_scores = MMR_LAMBDA * relevance_scores.copy()

            if len(mmr_selected) > 0:  # Only apply diversity when we have prior selections
                # Get vectors of all already-selected items
                selected_vecs = vecs_n[mmr_selected]  # [num_selected, emb_dim]

                # Compute similarity matrix: [num_candidates, num_selected]
                similarity_matrix = cand_vecs @ selected_vecs.T

                # Get max similarity for each candidate
                max_similarities = similarity_matrix.max(axis=1)

                # Update MMR scores with diversity penalty
                mmr_scores -= (1 - MMR_LAMBDA) * max_similarities

            # Mask out already-selected candidates
            mmr_scores[~remaining_mask] = -np.inf

            # Select candidate with highest MMR score
            best_idx = mmr_scores.argmax()
            selected_chunk_idx = cand_array[best_idx]

            mmr_selected.append(int(selected_chunk_idx))
            remaining_mask[best_idx] = False

    return mmr_selected


def apply_reranking(
    question: str,
    chunks: List[Dict],
    mmr_selected: List[int],
    scores: Dict[str, Any],
    use_rerank: bool,
    seed: int = DEFAULT_SEED,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    retries: int = DEFAULT_RETRIES
) -> Tuple[List[int], Dict, bool, str, float]:
    """Apply optional LLM reranking to MMR-selected chunks.

    Args:
        question: User question
        chunks: All chunks
        mmr_selected: List of MMR-selected chunk indices
        scores: Dict with relevance scores
        use_rerank: Whether to apply reranking
        seed, num_ctx, num_predict, retries: LLM parameters

    Returns:
        Tuple of (reranked_chunks, rerank_scores, rerank_applied, rerank_reason, timing)
    """
    rerank_scores = {}
    rerank_applied = False
    rerank_reason = "disabled"
    timing = 0.0

    if use_rerank:
        logger.debug(json.dumps({"event": "rerank_start", "candidates": len(mmr_selected)}))
        t0 = time.time()
        mmr_selected, rerank_scores, rerank_applied, rerank_reason = rerank_with_llm(
            question, chunks, mmr_selected, scores, seed=seed, num_ctx=num_ctx, num_predict=num_predict, retries=retries
        )
        timing = time.time() - t0
        logger.debug(json.dumps({"event": "rerank_done", "selected": len(mmr_selected), "scored": len(rerank_scores)}))

        # Add greppable rerank fallback log
        if not rerank_applied:
            logger.debug("info: rerank=fallback reason=%s", rerank_reason)

    return mmr_selected, rerank_scores, rerank_applied, rerank_reason, timing


def extract_citations(text: str) -> List[str]:
    """Extract citation IDs from answer text.

    Supports formats:
    - Single: [id_123], [123], [abc123-def]
    - Comma-separated: [id_a, id_b], [123, 456]
    - Mixed: [id_123, 456, abc-def]
    """
    import re
    # Match brackets containing citation IDs (single or comma-separated)
    # First, find all bracketed content: [...]
    bracket_pattern = r'\[([^\]]+)\]'
    bracket_matches = re.findall(bracket_pattern, text)

    citations = []
    for match in bracket_matches:
        # Split by comma and extract individual IDs
        # Match alphanumeric IDs with underscores and hyphens
        id_pattern = r'([a-zA-Z0-9_-]+)'
        ids = re.findall(id_pattern, match)
        citations.extend([id.strip() for id in ids if id.strip()])

    return citations


def validate_citations(answer: str, valid_chunk_ids: List) -> Tuple[bool, List[str], List[str]]:
    """Validate that citations in answer reference valid chunk IDs.

    Args:
        answer: Answer text with citations
        valid_chunk_ids: List of valid chunk IDs from context

    Returns:
        Tuple of (is_valid, valid_citations, invalid_citations)
    """
    extracted = extract_citations(answer)
    # Normalize to strings for comparison
    valid_set = set(str(cid) for cid in valid_chunk_ids)

    valid_citations = [cid for cid in extracted if cid in valid_set]
    invalid_citations = [cid for cid in extracted if cid not in valid_set]

    is_valid = len(invalid_citations) == 0
    return is_valid, valid_citations, invalid_citations


def generate_llm_answer(
    question: str,
    context_block: str,
    seed: int = DEFAULT_SEED,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    retries: int = DEFAULT_RETRIES,
    packed_ids: Optional[List] = None
) -> Tuple[str, float, Optional[int]]:
    """Generate answer from LLM given question and context with confidence scoring and citation validation.

    Args:
        question: User question
        context_block: Packed context snippets
        seed, num_ctx, num_predict, retries: LLM parameters
        packed_ids: List of chunk IDs included in context (for citation validation)

    Returns:
        Tuple of (answer_text, timing, confidence)
        - answer_text: The LLM's response
        - timing: Time taken for LLM call
        - confidence: 0-100 score, or None if not provided/parsable
    """
    t0 = time.time()
    raw_response = ask_llm(question, context_block, seed=seed, num_ctx=num_ctx, num_predict=num_predict, retries=retries).strip()
    timing = time.time() - t0

    # Parse JSON response with confidence
    confidence = None
    answer = raw_response  # Default to raw response if parsing fails

    try:
        # Try to parse as JSON
        # Handle markdown code blocks (```json ... ```)
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            # Extract content between ``` markers
            lines = cleaned.split("\n")
            # Remove first line (```json or ```) and last line (```)
            if len(lines) >= 3 and lines[-1].strip() == "```":
                cleaned = "\n".join(lines[1:-1]).strip()
            elif len(lines) >= 2:
                # Sometimes the closing ``` is on same line
                cleaned = "\n".join(lines[1:]).replace("```", "").strip()

        parsed = json.loads(cleaned)

        if isinstance(parsed, dict):
            answer = parsed.get("answer", raw_response)
            confidence = parsed.get("confidence")

            # Validate confidence is in 0-100 range
            if confidence is not None:
                try:
                    confidence = int(confidence)
                    if not (0 <= confidence <= 100):
                        logger.warning(f"Confidence out of range: {confidence}, ignoring")
                        confidence = None
                except (ValueError, TypeError):
                    logger.warning(f"Invalid confidence value: {confidence}, ignoring")
                    confidence = None
        else:
            answer = raw_response

    except json.JSONDecodeError:
        # Not JSON, use raw response
        answer = raw_response

    # Citation validation
    if packed_ids:
        has_citations = bool(extract_citations(answer))

        if not has_citations and answer != REFUSAL_STR:
            if STRICT_CITATIONS:
                logger.warning("Answer lacks citations in strict mode, refusing answer")
                answer = REFUSAL_STR
                confidence = None
            else:
                logger.warning("Answer lacks citations (expected format: [id_123, id_456])")

        # Validate citations reference actual chunks (only if not already refused)
        if answer != REFUSAL_STR:
            is_valid, valid_cites, invalid_cites = validate_citations(answer, packed_ids)

            if invalid_cites:
                if STRICT_CITATIONS:
                    logger.warning(f"Answer contains invalid citations in strict mode: {invalid_cites}, refusing answer")
                    answer = REFUSAL_STR
                    confidence = None
                else:
                    logger.warning(f"Answer contains invalid citations: {invalid_cites}")

    return answer, timing, confidence


def answer_once(
    question: str,
    chunks: List[Dict],
    vecs_n: np.ndarray,
    bm: Dict,
    hnsw=None,
    top_k: int = DEFAULT_TOP_K,
    pack_top: int = DEFAULT_PACK_TOP,
    threshold: float = DEFAULT_THRESHOLD,
    use_rerank: bool = False,
    seed: int = DEFAULT_SEED,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    retries: int = DEFAULT_RETRIES,
    faiss_index_path: Optional[str] = None
) -> Dict[str, Any]:
    """Complete answer generation pipeline.

    Args:
        question: User question
        chunks: List of all chunks
        vecs_n: Normalized embedding vectors
        bm: BM25 index
        hnsw: Optional HNSW index
        top_k: Number of candidates to retrieve
        pack_top: Number of chunks to pack in context
        threshold: Minimum similarity threshold
        use_rerank: Whether to apply LLM reranking
        seed, num_ctx, num_predict, retries: LLM parameters
        faiss_index_path: Path to FAISS index file

    Returns:
        Dict with answer and metadata
    """
    t_start = time.time()
    increment_counter(MetricNames.QUERIES_TOTAL)

    # Retrieve
    t0 = time.time()
    selected, scores = retrieve(
        question, chunks, vecs_n, bm,
        top_k=top_k, hnsw=hnsw, retries=retries,
        faiss_index_path=faiss_index_path
    )
    retrieve_time = time.time() - t0
    observe_histogram(MetricNames.RETRIEVAL_LATENCY, retrieve_time * 1000.0)

    # Check coverage
    if not coverage_ok(selected, scores["dense"], threshold):
        total_ms = (time.time() - t_start) * 1000.0
        observe_histogram(MetricNames.QUERY_LATENCY, total_ms)
        increment_counter(MetricNames.REFUSALS_TOTAL, labels={"stage": "coverage"})
        return {
            "answer": REFUSAL_STR,
            "refused": True,
            "confidence": None,
            "selected_chunks": [],
            "selected_chunk_ids": [],
            "packed_chunks": [],
            "packed_chunk_ids": [],
            "context_block": "",
            "timing": {
                "total_ms": (time.time() - t_start) * 1000,
                "retrieve_ms": retrieve_time * 1000,
                "mmr_ms": 0,
                "rerank_ms": 0,
                "llm_ms": 0,
            },
            "metadata": {
                "retrieval_count": len(selected),
                "coverage_check": "failed"
            }
        }

    # Apply MMR diversification
    t0 = time.time()
    mmr_selected = apply_mmr_diversification(selected, scores, vecs_n, pack_top)
    mmr_time = time.time() - t0

    # Optional reranking
    mmr_selected, rerank_scores, rerank_applied, rerank_reason, rerank_time = apply_reranking(
        question, chunks, mmr_selected, scores, use_rerank,
        seed=seed, num_ctx=num_ctx, num_predict=num_predict, retries=retries
    )

    # Pack snippets
    context_block, packed_ids, used_tokens = pack_snippets(
        chunks, mmr_selected, pack_top=pack_top, num_ctx=num_ctx
    )

    selected_chunk_ids = _resolve_chunk_ids(chunks, selected)
    packed_chunk_ids = list(packed_ids or [])

    # Generate answer
    answer, llm_time, confidence = generate_llm_answer(
        question, context_block,
        seed=seed, num_ctx=num_ctx, num_predict=num_predict,
        retries=retries, packed_ids=packed_ids
    )

    total_time = time.time() - t_start

    # OPTIMIZATION (Analysis Section 9.1 #4): Confidence-based routing
    # Auto-escalate low-confidence queries to human review
    refused = (answer == REFUSAL_STR)
    routing = get_routing_action(confidence, refused=refused, critical=False)

    observe_histogram(MetricNames.RERANK_LATENCY, rerank_time * 1000.0)
    observe_histogram(MetricNames.LLM_LATENCY, llm_time * 1000.0)
    observe_histogram(MetricNames.QUERY_LATENCY, total_time * 1000.0)

    if refused:
        increment_counter(MetricNames.REFUSALS_TOTAL, labels={"stage": "llm"})

    return {
        "answer": answer,
        "refused": refused,
        "confidence": confidence,
        "selected_chunks": selected,
        "selected_chunk_ids": selected_chunk_ids,
        "packed_chunks": mmr_selected,
        "packed_chunk_ids": packed_chunk_ids,
        "context_block": context_block,
        "timing": {
            "total_ms": total_time * 1000,
            "retrieve_ms": retrieve_time * 1000,
            "mmr_ms": mmr_time * 1000,
            "rerank_ms": rerank_time * 1000,
            "llm_ms": llm_time * 1000,
        },
        "metadata": {
            "retrieval_count": len(selected),
            "packed_count": len(packed_ids),
            "used_tokens": used_tokens,
            "rerank_applied": rerank_applied,
            "rerank_reason": rerank_reason,
        },
        "routing": routing  # Add routing recommendation
    }


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
    from .config import KPI, EMB_BACKEND, USE_ANN, ALPHA_HYBRID

    budget_tokens = 0 if used_tokens is None else int(used_tokens)
    result = {
        "answer": answer,
        "citations": citations,
        "debug": {
            "meta": {
                "used_tokens": budget_tokens,
                "topk": topk,
                "packed": packed,
                "emb_backend": EMB_BACKEND,
                "ann": USE_ANN,
                "alpha": ALPHA_HYBRID
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


__all__ = [
    "apply_mmr_diversification",
    "apply_reranking",
    "extract_citations",
    "validate_citations",
    "generate_llm_answer",
    "answer_once",
    "answer_to_json",
]
