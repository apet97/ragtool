"""Async support for non-blocking Ollama API calls.

OPTIMIZATION (Analysis Section 9.1 #1): Async LLM calls for 2-4x concurrent throughput.
This module provides async versions of HTTP and LLM functions while maintaining
backward compatibility with the synchronous API.

Usage:
    # Async mode (requires asyncio event loop)
    import asyncio
    from clockify_rag.async_support import async_answer_once

    result = asyncio.run(async_answer_once(question, chunks, vecs_n, bm))

    # Synchronous mode (default, no changes needed)
    from clockify_rag.answer import answer_once

    result = answer_once(question, chunks, vecs_n, bm)
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Tuple, Optional, Any

import aiohttp
import numpy as np

from .config import (
    OLLAMA_URL,
    GEN_MODEL,
    EMB_MODEL,
    EMB_CONNECT_T,
    EMB_READ_T,
    CHAT_CONNECT_T,
    CHAT_READ_T,
    DEFAULT_TOP_K,
    DEFAULT_PACK_TOP,
    DEFAULT_THRESHOLD,
    DEFAULT_SEED,
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_PREDICT,
    DEFAULT_RETRIES,
    REFUSAL_STR,
)
from .exceptions import LLMError
from .confidence_routing import get_routing_action

logger = logging.getLogger(__name__)


async def async_http_post_with_retries(
    session: aiohttp.ClientSession,
    url: str,
    json_payload: dict,
    retries: int = 3,
    timeout: Optional[aiohttp.ClientTimeout] = None
) -> dict:
    """Async POST with exponential backoff retry.

    Args:
        session: aiohttp ClientSession
        url: Target URL
        json_payload: JSON payload to POST
        retries: Number of retries
        timeout: Request timeout

    Returns:
        JSON response

    Raises:
        aiohttp.ClientError: If request fails after all retries
    """
    if timeout is None:
        timeout = aiohttp.ClientTimeout(
            connect=EMB_CONNECT_T,
            total=EMB_READ_T + EMB_CONNECT_T
        )

    last_error = None
    for attempt in range(retries + 1):
        try:
            async with session.post(url, json=json_payload, timeout=timeout) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            last_error = e
            if attempt < retries:
                backoff = 0.5 * (2 ** attempt)  # Exponential backoff
                logger.debug(f"HTTP POST failed (attempt {attempt + 1}/{retries + 1}), retrying in {backoff}s: {e}")
                await asyncio.sleep(backoff)
            else:
                logger.error(f"HTTP POST to {url} failed after {retries + 1} attempts: {e}")

    raise aiohttp.ClientError(f"HTTP POST to {url} failed after {retries + 1} attempts: {last_error}")


async def async_embed_query(session: aiohttp.ClientSession, text: str, retries: int = 0) -> np.ndarray:
    """Async version of embed_query.

    Args:
        session: aiohttp ClientSession
        text: Text to embed
        retries: Number of retries

    Returns:
        Normalized embedding vector (numpy array)
    """
    payload = {"model": EMB_MODEL, "prompt": text}
    timeout = aiohttp.ClientTimeout(
        connect=EMB_CONNECT_T,
        total=EMB_READ_T + EMB_CONNECT_T
    )

    try:
        result = await async_http_post_with_retries(
            session,
            f"{OLLAMA_URL}/api/embeddings",
            payload,
            retries=retries,
            timeout=timeout
        )
        embedding = np.array(result["embedding"], dtype=np.float32)
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        raise LLMError(f"Embedding failed: {e}") from e


async def async_ask_llm(
    session: aiohttp.ClientSession,
    question: str,
    context_block: str,
    seed: int = DEFAULT_SEED,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    retries: int = DEFAULT_RETRIES
) -> str:
    """Async version of ask_llm.

    Args:
        session: aiohttp ClientSession
        question: User question
        context_block: Context snippets
        seed, num_ctx, num_predict, retries: LLM parameters

    Returns:
        LLM response text
    """
    from .retrieval import get_system_prompt, USER_WRAPPER

    system_prompt = get_system_prompt()
    user_prompt = USER_WRAPPER.format(snips=context_block, q=question)

    payload = {
        "model": GEN_MODEL,
        "prompt": user_prompt,
        "system": system_prompt,
        "options": {
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
        "stream": False,
    }

    timeout = aiohttp.ClientTimeout(
        connect=CHAT_CONNECT_T,
        total=CHAT_READ_T + CHAT_CONNECT_T
    )

    try:
        result = await async_http_post_with_retries(
            session,
            f"{OLLAMA_URL}/api/generate",
            payload,
            retries=retries,
            timeout=timeout
        )
        return result.get("response", "")
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise LLMError(f"LLM generation failed: {e}") from e


async def async_generate_llm_answer(
    session: aiohttp.ClientSession,
    question: str,
    context_block: str,
    seed: int = DEFAULT_SEED,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    retries: int = DEFAULT_RETRIES,
    packed_ids: Optional[List] = None
) -> Tuple[str, float, Optional[int]]:
    """Async version of generate_llm_answer with confidence scoring and citation validation.

    Args:
        session: aiohttp ClientSession
        question: User question
        context_block: Packed context snippets
        seed, num_ctx, num_predict, retries: LLM parameters
        packed_ids: List of chunk IDs included in context (for citation validation)

    Returns:
        Tuple of (answer_text, timing, confidence)
    """
    from .answer import extract_citations, validate_citations
    from .config import STRICT_CITATIONS

    t0 = time.time()
    raw_response = (await async_ask_llm(session, question, context_block, seed, num_ctx, num_predict, retries)).strip()
    timing = time.time() - t0

    # Parse JSON response with confidence
    confidence = None
    answer = raw_response  # Default to raw response if parsing fails

    try:
        # Try to parse as JSON
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            # Extract content between ``` markers
            lines = cleaned.split("\n")
            if len(lines) >= 3 and lines[-1].strip() == "```":
                cleaned = "\n".join(lines[1:-1]).strip()
            elif len(lines) >= 2:
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


async def async_answer_once(
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
    """Async version of answer_once for non-blocking LLM calls.

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
        Dict with answer and metadata (same format as answer_once)
    """
    from .retrieval import retrieve, coverage_ok, pack_snippets
    from .answer import apply_mmr_diversification

    t_start = time.time()

    # Create aiohttp session for async HTTP calls
    async with aiohttp.ClientSession() as session:
        # Retrieve (synchronous for now, retrieval is fast)
        t0 = time.time()
        from .retrieval import retrieve
        selected, scores = retrieve(
            question, chunks, vecs_n, bm,
            top_k=top_k, hnsw=hnsw, retries=retries,
            faiss_index_path=faiss_index_path
        )
        retrieve_time = time.time() - t0

        # Check coverage
        if not coverage_ok(selected, scores["dense"], threshold):
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
                },
                "routing": get_routing_action(None, refused=True, critical=False)
            }

        # Apply MMR diversification
        t0 = time.time()
        mmr_selected = apply_mmr_diversification(selected, scores, vecs_n, pack_top)
        mmr_time = time.time() - t0

        # Reranking (synchronous for now)
        rerank_time = 0.0
        rerank_applied = False
        rerank_reason = "disabled"
        if use_rerank:
        from .answer import apply_reranking
            t0 = time.time()
            mmr_selected, rerank_scores, rerank_applied, rerank_reason, rerank_time = apply_reranking(
                question, chunks, mmr_selected, scores, use_rerank,
                seed=seed, num_ctx=num_ctx, num_predict=num_predict, retries=retries
            )

        # Pack snippets
        context_block, packed_ids, used_tokens = pack_snippets(
            chunks, mmr_selected, pack_top=pack_top, num_ctx=num_ctx
        )

        # Generate answer (async)
        answer, llm_time, confidence = await async_generate_llm_answer(
            session, question, context_block,
            seed=seed, num_ctx=num_ctx, num_predict=num_predict,
            retries=retries, packed_ids=packed_ids
        )

        from .answer import _resolve_chunk_ids as _resolve_chunk_ids_sync

        selected_chunk_ids = _resolve_chunk_ids_sync(chunks, selected)
        packed_chunk_ids = list(packed_ids or [])

        total_time = time.time() - t_start

        # Confidence-based routing
        refused = (answer == REFUSAL_STR)
        routing = get_routing_action(confidence, refused=refused, critical=False)

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
            "routing": routing
        }


__all__ = [
    "async_http_post_with_retries",
    "async_embed_query",
    "async_ask_llm",
    "async_generate_llm_answer",
    "async_answer_once",
]
