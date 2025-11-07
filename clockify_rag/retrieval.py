"""Retrieval module for Clockify RAG system.

This module contains all retrieval-related functionality:
- Query expansion and embedding
- Hybrid retrieval (BM25 + dense + MMR)
- LLM-based reranking
- Snippet packing with token budget management
- Coverage checking
- Answer generation with LLM
"""

import json
import logging
import math
import os
import pathlib
import re
import time
import unicodedata
from collections import Counter
from typing import Any, Optional, Dict, List, Tuple

import numpy as np
import requests

import clockify_rag.config as config
from .embedding import embed_query as _embedding_embed_query
from .exceptions import LLMError
from .http_utils import get_session
from .indexing import bm25_scores, load_faiss_index

logger = logging.getLogger(__name__)

# ====== GLOBAL STATE FOR FAISS ======
_FAISS_INDEX = None
_FAISS_INDEX_LOCK = __import__('threading').RLock()

# ====== RETRIEVAL PROFILING ======
# FIX: Add thread-safe lock for concurrent access to profiling state
_RETRIEVE_PROFILE_LOCK = __import__('threading').RLock()
RETRIEVE_PROFILE_LAST = {}

# ====== PROMPTS ======
_SYSTEM_PROMPT_TEMPLATE = """You are CAKE.com Internal Support for Clockify.
Closed-book. Only use SNIPPETS. If info is missing, reply exactly "{refusal}" and set confidence to 0.
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
- Ensure the entire output remains valid JSON with no extra prose or markdown wrappers."""


def get_system_prompt() -> str:
    """Return the system prompt with the current refusal string.

    This function allows runtime configuration of the refusal string,
    enabling dynamic prompt generation for testing and customization.
    """
    return _SYSTEM_PROMPT_TEMPLATE.format(refusal=config.REFUSAL_STR)

USER_WRAPPER = """SNIPPETS:
{snips}

QUESTION:
{q}

Respond with only a JSON object following the schema {{"answer": "...", "confidence": 0-100}}.
Keep all narrative content inside the answer field and include citations as described in the system message.
Do not add markdown fences or text outside the JSON object."""

RERANK_PROMPT = """You rank passages for a Clockify support answer. Score each 0.0–1.0 strictly.
Output JSON only: [{{"id":"<chunk_id>","score":0.82}}, ...].

QUESTION:
{q}

PASSAGES:
{passages}"""

# ====== QUERY EXPANSION ======
QUERY_EXPANSIONS_ENV_VAR = "CLOCKIFY_QUERY_EXPANSIONS"
_DEFAULT_QUERY_EXPANSION_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "query_expansions.json"
_query_expansion_cache = None
_query_expansion_override = None


def set_query_expansion_path(path):
    """Override the query expansion configuration file path."""
    global _query_expansion_override
    if path is None:
        _query_expansion_override = None
    else:
        _query_expansion_override = pathlib.Path(path)
    reset_query_expansion_cache()


def reset_query_expansion_cache():
    """Clear cached query expansion data (useful for tests)."""
    global _query_expansion_cache
    _query_expansion_cache = None


def _resolve_query_expansion_path():
    if _query_expansion_override is not None:
        return _query_expansion_override
    env_path = os.environ.get(QUERY_EXPANSIONS_ENV_VAR)
    if env_path:
        return pathlib.Path(env_path)
    return _DEFAULT_QUERY_EXPANSION_PATH


def _read_query_expansion_file(path):
    MAX_EXPANSION_FILE_SIZE = int(os.environ.get("MAX_QUERY_EXPANSION_FILE_SIZE", str(10 * 1024 * 1024)))
    try:
        file_size = os.path.getsize(path)
        if file_size > MAX_EXPANSION_FILE_SIZE:
            raise ValueError(
                f"Query expansion file too large ({file_size} bytes, max {MAX_EXPANSION_FILE_SIZE}). "
                f"Set MAX_QUERY_EXPANSION_FILE_SIZE env var to override."
            )

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise ValueError(f"Query expansion file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in query expansion file {path}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to read query expansion file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Query expansion config must be a JSON object (file: {path})")

    normalized = {}
    for term, synonyms in data.items():
        if not isinstance(term, str):
            raise ValueError(f"Query expansion terms must be strings (file: {path})")
        if not isinstance(synonyms, list):
            raise ValueError(f"Query expansion entry for '{term}' must be a list (file: {path})")

        cleaned = []
        for syn in synonyms:
            syn_str = syn if isinstance(syn, str) else str(syn)
            syn_str = syn_str.strip()
            if syn_str:
                cleaned.append(syn_str)

        if cleaned:
            normalized[term.lower()] = cleaned

    return normalized


def load_query_expansion_dict(force_reload=False, suppress_errors=True):
    """Load query expansion dictionary from disk with optional caching."""
    global _query_expansion_cache

    if not force_reload and _query_expansion_cache is not None:
        return _query_expansion_cache

    path = _resolve_query_expansion_path()

    try:
        _query_expansion_cache = _read_query_expansion_file(path)
        return _query_expansion_cache
    except Exception as e:
        if suppress_errors:
            logger.warning(f"Failed to load query expansion config from {path}: {e}")
            _query_expansion_cache = {}
            return _query_expansion_cache
        else:
            raise


def tokenize(s: str) -> List[str]:
    """Simple tokenizer: lowercase [a-z0-9]+."""
    s = s.lower()
    s = unicodedata.normalize("NFKC", s)
    return re.findall(r"[a-z0-9]+", s)


def approx_tokens(chars: int) -> int:
    """Estimate tokens: 1 token ≈ 4 chars."""
    return max(1, chars // 4)


def count_tokens(text: str, model: str = None) -> int:
    """Count actual tokens using tiktoken for supported models.

    Falls back to model-specific heuristics for Qwen and other models.

    Args:
        text: Text to count tokens for
        model: Model name (defaults to config.GEN_MODEL)

    Returns:
        Estimated token count
    """
    from . import config

    if model is None:
        model = config.GEN_MODEL

    # Try tiktoken for GPT models
    if "gpt" in model.lower():
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except (ImportError, KeyError):
            pass

    # For Qwen and other models, use improved heuristic
    # Qwen tokenizer tends to be more efficient than GPT for English (~3.5 chars/token)
    # but less efficient for CJK content (~1.5 chars/token)
    if "qwen" in model.lower():
        # Count CJK characters (Chinese, Japanese, Korean)
        import re
        cjk_pattern = r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]'
        cjk_chars = len(re.findall(cjk_pattern, text))
        non_cjk_chars = len(text) - cjk_chars

        # CJK: ~1.5 chars/token, non-CJK: ~3.5 chars/token
        # Use ceil to ensure we never underestimate (safer for budget enforcement)
        return math.ceil(cjk_chars / 1.5 + non_cjk_chars / 3.5)

    # Default fallback
    return approx_tokens(len(text))


def truncate_to_token_budget(text: str, budget: int) -> str:
    """Truncate text to fit token budget, append ellipsis."""
    est_tokens = count_tokens(text)
    if est_tokens <= budget:
        return text

    # Binary search for optimal truncation point
    left, right = 0, len(text)
    ellipsis = "..."
    ellipsis_tokens = count_tokens(ellipsis)
    target = budget - ellipsis_tokens

    while left < right:
        mid = (left + right + 1) // 2
        candidate = text[:mid]
        if count_tokens(candidate) <= target:
            left = mid
        else:
            right = mid - 1

    return text[:left] + ellipsis


def expand_query(question: str) -> str:
    """Expand query with domain-specific synonyms and acronyms.

    Returns expanded query string with original + synonym terms.
    Example: "How to track time?" → "How to track log record enter time hours duration?"
    """
    if not question:
        return question

    expansions = load_query_expansion_dict()
    q_lower = question.lower()
    expanded_terms = []

    for term, synonyms in expansions.items():
        if re.search(r'\b' + re.escape(term) + r'\b', q_lower):
            for syn in synonyms:
                if syn not in expanded_terms:
                    expanded_terms.append(syn)

    if expanded_terms:
        expansion = " ".join(expanded_terms)
        return f"{question} {expansion}"

    return question


def normalize_scores_zscore(arr: np.ndarray) -> np.ndarray:
    """Z-score normalize."""
    a = np.asarray(arr, dtype="float32")
    if a.size == 0:
        return a
    m, s = a.mean(), a.std()
    if s == 0:
        return a  # Preserve original when no variance
    return (a - m) / s


def embed_query(question: str, retries=0) -> np.ndarray:
    """Embed a query using the configured backend.

    Delegates to :mod:`clockify_rag.embedding` so the query vector shares the
    same dimensionality and normalization strategy as stored document
    embeddings, regardless of whether the backend is Ollama or the local
    SentenceTransformer.
    """

    return _embedding_embed_query(question, retries=retries)


class DenseScoreStore:
    """Container for dense similarity scores with optional lazy materialization."""

    __slots__ = ("_length", "_full", "_vecs", "_qv", "_cache")

    def __init__(self, length: int, *, full_scores: Optional[np.ndarray] = None,
                 vecs: Optional[np.ndarray] = None, qv: Optional[np.ndarray] = None,
                 initial: Optional[list] = None) -> None:
        self._length = int(length)
        self._full: Optional[np.ndarray] = None
        self._vecs = vecs
        self._qv = qv
        self._cache: dict = {}

        if full_scores is not None:
            self._full = np.asarray(full_scores, dtype="float32")
        elif initial:
            self._cache.update({int(idx): float(score) for idx, score in initial})

    def __len__(self) -> int:
        return self._length

    def _materialize_full(self) -> np.ndarray:
        if self._full is None:
            if self._vecs is None or self._qv is None:
                self._full = np.zeros(self._length, dtype="float32")
            else:
                self._full = self._vecs.dot(self._qv).astype("float32")
        return self._full

    def __getitem__(self, idx: int) -> float:
        idx = int(idx)
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)

        if self._full is not None:
            return float(self._full[idx])

        if idx not in self._cache:
            if self._vecs is None or self._qv is None:
                raise KeyError(idx)
            self._cache[idx] = float(self._vecs[idx].dot(self._qv))
        return self._cache[idx]

    def get(self, idx: int, default: Optional[float] = None) -> Optional[float]:
        try:
            return self[idx]
        except (IndexError, KeyError):
            return default

    def to_array(self) -> np.ndarray:
        return self._materialize_full().copy()


def retrieve(question: str, chunks, vecs_n, bm, top_k=12, hnsw=None, retries=0,
             faiss_index_path=None) -> Tuple[List[int], Dict[str, Any]]:
    """Hybrid retrieval: dense + BM25 + dedup. Optionally uses FAISS/HNSW for fast K-NN.

    Scoring: hybrid = config.ALPHA_HYBRID * normalize(BM25) + (1 - config.ALPHA_HYBRID) * normalize(dense)

    Query expansion: Applies domain-specific synonym expansion for BM25 (keyword-based),
    uses original query for dense retrieval (embeddings already capture semantics).

    Returns:
        Tuple of (filtered_indices, scores_dict) where filtered_indices is list of int
        and scores_dict contains 'dense', 'bm25', and 'hybrid' numpy arrays.
    """
    global _FAISS_INDEX, RETRIEVE_PROFILE_LAST

    # Expand query for BM25 keyword matching
    expanded_question = expand_query(question)

    # Use original question for embedding
    qv_n = embed_query(question, retries=retries)

    # Try to load FAISS index once on first call
    if config.USE_ANN == "faiss" and _FAISS_INDEX is None:
        with _FAISS_INDEX_LOCK:
            if _FAISS_INDEX is None and faiss_index_path:
                _FAISS_INDEX = load_faiss_index(faiss_index_path)
                if _FAISS_INDEX:
                    # Only set nprobe for IVF indexes (not flat indexes)
                    if hasattr(_FAISS_INDEX, 'nprobe'):
                        _FAISS_INDEX.nprobe = config.ANN_NPROBE
                    logger.info("info: ann=faiss status=loaded nprobe=%d", config.ANN_NPROBE)
                else:
                    logger.info("info: ann=fallback reason=missing-index")

    dense_scores_full = None
    candidate_idx: List[int] = []
    n_chunks = len(chunks)
    dot_elapsed = 0.0
    dense_computed = 0

    if _FAISS_INDEX:
        # Only score FAISS candidates, don't compute full corpus
        D, I = _FAISS_INDEX.search(
            qv_n.reshape(1, -1).astype("float32"),
            max(config.ANN_CANDIDATE_MIN, top_k * config.FAISS_CANDIDATE_MULTIPLIER),
        )
        # Filter indices and distances together to maintain alignment
        # (prevents misalignment when FAISS returns -1 sentinels)
        valid_pairs = [(int(i), float(d)) for i, d in zip(I[0], D[0]) if 0 <= i < n_chunks]
        candidate_idx = [i for i, _ in valid_pairs]
        dense_from_ann = np.array([d for _, d in valid_pairs], dtype=np.float32)

        dense_scores = dense_from_ann
        dense_scores_full = None
        dense_computed = len(candidate_idx)
        dot_elapsed = 0.0
    elif hnsw:
        _, cand = hnsw.knn_query(qv_n, k=max(config.ANN_CANDIDATE_MIN, top_k * config.FAISS_CANDIDATE_MULTIPLIER))
        candidate_idx = cand[0].tolist()
        dot_start = time.perf_counter()
        dense_scores_full = vecs_n.dot(qv_n)
        dot_elapsed = time.perf_counter() - dot_start
        dense_computed = n_chunks
    else:
        dot_start = time.perf_counter()
        dense_scores_full = vecs_n.dot(qv_n)
        dot_elapsed = time.perf_counter() - dot_start
        dense_computed = n_chunks
        dense_scores = dense_scores_full
        candidate_idx = np.arange(len(chunks)).tolist()

    if not candidate_idx:
        dense_scores_full = vecs_n.dot(qv_n)
        max_candidates = max(config.ANN_CANDIDATE_MIN, top_k * config.FAISS_CANDIDATE_MULTIPLIER)
        if len(chunks) > max_candidates:
            top_indices = np.argsort(dense_scores_full)[::-1][:max_candidates]
            candidate_idx = top_indices.tolist()
        else:
            candidate_idx = np.arange(len(chunks)).tolist()
        dense_scores = dense_scores_full

    candidate_idx_array = np.array(candidate_idx, dtype=np.int32)

    # Use expanded query for BM25
    bm_scores_full = bm25_scores(expanded_question, bm, top_k=top_k * 3)

    # Normalize once, then slice for candidates
    zs_bm_full = normalize_scores_zscore(bm_scores_full)
    zs_dense_full = None
    if dense_scores_full is not None:
        dense_scores_full = np.asarray(dense_scores_full, dtype="float32")
        zs_dense_full = normalize_scores_zscore(dense_scores_full)
        zs_dense = zs_dense_full[candidate_idx_array] if candidate_idx_array.size else np.array([], dtype="float32")
    else:
        dense_scores = np.asarray(dense_scores, dtype="float32")
        zs_dense = normalize_scores_zscore(dense_scores)
    zs_bm = zs_bm_full[candidate_idx_array] if candidate_idx_array.size else np.array([], dtype="float32")

    # Hybrid scoring
    hybrid = config.ALPHA_HYBRID * zs_bm + (1 - config.ALPHA_HYBRID) * zs_dense
    if hybrid.size:
        top_positions = np.argsort(hybrid)[::-1][:top_k]
        top_idx = candidate_idx_array[top_positions]
    else:
        top_idx = np.array([], dtype=np.int32)

    # Deduplication
    seen = set()
    filtered = []
    for i in top_idx:
        key = (chunks[i]["title"], chunks[i]["section"])
        if key in seen:
            continue
        seen.add(key)
        filtered.append(i)

    # Reuse cached normalized scores for full hybrid
    if zs_dense_full is not None:
        hybrid_full = config.ALPHA_HYBRID * zs_bm_full + (1 - config.ALPHA_HYBRID) * zs_dense_full
    else:
        hybrid_full = np.zeros(len(chunks), dtype="float32")
        for idx, score in zip(candidate_idx, hybrid):
            hybrid_full[idx] = score

    if dense_scores_full is not None:
        dense_scores_store = DenseScoreStore(len(chunks), full_scores=dense_scores_full)
    else:
        dense_scores_store = DenseScoreStore(
            len(chunks), vecs=vecs_n, qv=qv_n,
            initial=list(zip(candidate_idx, dense_scores))
        )

    dense_total = n_chunks
    used_hnsw = bool(hnsw) and _FAISS_INDEX is None
    dense_computed_total = dense_computed or (dense_total if (used_hnsw or not _FAISS_INDEX) else 0)
    dense_reused = dense_total - dense_computed_total

    # FIX: Thread-safe update of profiling state
    global RETRIEVE_PROFILE_LAST
    profile_data = {
        "used_faiss": bool(_FAISS_INDEX),
        "used_hnsw": used_hnsw,
        "candidates": int(len(candidate_idx)),
        "dense_total": int(dense_total),
        "dense_reused": int(dense_reused),
        "dense_computed": int(dense_computed_total),
        "dense_saved": int(dense_total - dense_computed_total),
        "dense_dot_time_ms": round(dot_elapsed * 1000, 3),
    }

    with _RETRIEVE_PROFILE_LOCK:
        RETRIEVE_PROFILE_LAST = profile_data

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "profile: retrieval ann=%s reused=%d computed=%d total=%d dot_ms=%.3f",
            "faiss" if _FAISS_INDEX else ("hnsw" if used_hnsw else "linear"),
            profile_data["dense_reused"],
            profile_data["dense_computed"],
            dense_total,
            profile_data["dense_dot_time_ms"],
        )

    return filtered, {
        "dense": dense_scores_store,
        "bm25": bm_scores_full,
        "hybrid": hybrid_full
    }


def rerank_with_llm(
    question: str,
    chunks,
    selected,
    scores,
    seed: Optional[int] = None,
    num_ctx: Optional[int] = None,
    num_predict: Optional[int] = None,
    retries: Optional[int] = None,
) -> Tuple:
    """Optional: rerank MMR-selected passages with LLM.

    Returns: (order, scores, rerank_applied, rerank_reason)
    """
    if len(selected) <= 1:
        return selected, {}, False, "disabled"

    # Build passage list
    passages_text = "\n\n".join([
        f"[id={chunks[i]['id']}]\n{chunks[i]['text'][:config.RERANK_SNIPPET_MAX_CHARS]}"
        for i in selected
    ])
    if seed is None:
        seed = config.DEFAULT_SEED
    if num_ctx is None:
        num_ctx = config.DEFAULT_NUM_CTX
    if num_predict is None:
        num_predict = config.DEFAULT_NUM_PREDICT
    if retries is None:
        retries = config.DEFAULT_RETRIES

    payload = {
        "model": config.GEN_MODEL,
        "options": {
            "temperature": 0,
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.05
        },
        "messages": [
            {"role": "user", "content": RERANK_PROMPT.format(q=question, passages=passages_text)}
        ],
        "stream": False
    }

    rerank_scores = {}
    sess = get_session(retries=retries)
    try:
        r = sess.post(
            f"{config.OLLAMA_URL}/api/chat",
            json=payload,
            timeout=(config.CHAT_CONNECT_T, config.RERANK_READ_T),
            allow_redirects=False
        )
        r.raise_for_status()
        resp = r.json()
        msg = (resp.get("message") or {}).get("content", "").strip()

        if not msg:
            logger.debug("info: rerank=fallback reason=empty")
            return selected, rerank_scores, False, "empty"

        # Try to parse strict JSON array
        try:
            ranked = json.loads(msg)
            if not isinstance(ranked, list):
                logger.debug("info: rerank=fallback reason=json")
                return selected, rerank_scores, False, "json"

            # Map back to indices
            cid_to_idx = {chunks[i]["id"]: i for i in selected}
            reranked = []
            for entry in ranked:
                idx = cid_to_idx.get(entry.get("id"))
                if idx is not None:
                    score = entry.get("score", 0)
                    rerank_scores[idx] = score
                    reranked.append((idx, score))

            if reranked:
                reranked.sort(key=lambda x: x[1], reverse=True)
                return [idx for idx, _ in reranked], rerank_scores, True, ""
            else:
                logger.debug("info: rerank=fallback reason=empty")
                return selected, rerank_scores, False, "empty"
        except json.JSONDecodeError:
            logger.debug("info: rerank=fallback reason=json")
            return selected, rerank_scores, False, "json"
    except requests.exceptions.Timeout:
        logger.debug("info: rerank=fallback reason=timeout")
        return selected, rerank_scores, False, "timeout"
    except requests.exceptions.ConnectionError:
        logger.debug("info: rerank=fallback reason=conn")
        return selected, rerank_scores, False, "conn"
    except requests.exceptions.HTTPError:
        logger.debug(f"info: rerank=fallback reason=http")
        return selected, rerank_scores, False, "http"
    except requests.exceptions.RequestException:
        logger.debug("info: rerank=fallback reason=http")
        return selected, rerank_scores, False, "http"
    except Exception:
        logger.debug("info: rerank=fallback reason=http")
        return selected, rerank_scores, False, "http"


def _fmt_snippet_header(chunk):
    """Format chunk header: [id | title | section] + optional URL."""
    hdr = f"[{chunk['id']} | {chunk['title']} | {chunk['section']}]"
    if chunk.get("url"):
        hdr += f"\n{chunk['url']}"
    return hdr


def pack_snippets(
    chunks,
    order,
    pack_top: Optional[int] = None,
    budget_tokens: Optional[int] = None,
    num_ctx: Optional[int] = None,
):
    """Pack snippets respecting strict token budget and hard snippet cap.

    Guarantees:
    - Never exceeds min(config.CTX_TOKEN_BUDGET, num_ctx * 0.6)
    - Respects model's actual context window via num_ctx
    - First item always included (truncate body if needed; mark [TRUNCATED])
    - Returns (block, ids, used_tokens)
    """
    if pack_top is None:
        pack_top = config.DEFAULT_PACK_TOP
    if budget_tokens is None:
        budget_tokens = config.CTX_TOKEN_BUDGET
    if num_ctx is None:
        num_ctx = config.DEFAULT_NUM_CTX

    # Honor num_ctx: reserve 40% for system prompt + answer generation
    # Use the minimum of the configured budget and 60% of model's context window
    effective_budget = min(budget_tokens, int(num_ctx * 0.6))

    out = []
    ids = []
    used = 0
    first_truncated = False

    sep_text = "\n\n---\n\n"
    sep_tokens = count_tokens(sep_text)

    for idx_pos, idx in enumerate(order):
        if len(ids) >= pack_top:
            break

        c = chunks[idx]
        hdr = _fmt_snippet_header(c)
        body = c["text"]

        hdr_tokens = count_tokens(hdr + "\n")
        body_tokens = count_tokens(body)
        need_sep = 1 if out else 0
        sep_cost = sep_tokens if need_sep else 0

        if idx_pos == 0 and not ids:
            # Always include first; truncate if needed to fit budget
            item_tokens = hdr_tokens + body_tokens
            if item_tokens > effective_budget:
                allow_body = max(1, effective_budget - hdr_tokens)
                body = truncate_to_token_budget(body, allow_body)
                body_tokens = count_tokens(body)
                item_tokens = hdr_tokens + body_tokens
                first_truncated = True
            out.append(hdr + "\n" + body)
            ids.append(c["id"])
            used += item_tokens
            continue

        # For subsequent items, check sep + header + body within budget
        item_tokens = hdr_tokens + body_tokens
        if used + sep_cost + item_tokens <= effective_budget:
            if need_sep:
                out.append(sep_text)
            out.append(hdr + "\n" + body)
            ids.append(c["id"])
            used += sep_cost + item_tokens
        else:
            break

    if first_truncated and out:
        out[0] = out[0].replace("]", " [TRUNCATED]]", 1)

    return "".join(out), ids, used


def coverage_ok(selected, dense_scores, threshold):
    """Check coverage."""
    if len(selected) < config.COVERAGE_MIN_CHUNKS:
        return False
    highs = sum(1 for i in selected if dense_scores[i] >= threshold)
    return highs >= 2


def ask_llm(
    question: str,
    snippets_block: str,
    seed: Optional[int] = None,
    num_ctx: Optional[int] = None,
    num_predict: Optional[int] = None,
    retries: Optional[int] = None,
) -> str:
    """Call Ollama chat with Qwen best-practice options."""
    if seed is None:
        seed = config.DEFAULT_SEED
    if num_ctx is None:
        num_ctx = config.DEFAULT_NUM_CTX
    if num_predict is None:
        num_predict = config.DEFAULT_NUM_PREDICT
    if retries is None:
        retries = config.DEFAULT_RETRIES

    payload = {
        "model": config.GEN_MODEL,
        "options": {
            "temperature": 0,
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.05
        },
        "messages": [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": USER_WRAPPER.format(snips=snippets_block, q=question)}
        ],
        "stream": False
    }
    sess = get_session(retries=retries)

    try:
        r = sess.post(
            f"{config.OLLAMA_URL}/api/chat",
            json=payload,
            timeout=(config.CHAT_CONNECT_T, config.CHAT_READ_T),
            allow_redirects=False
        )
        r.raise_for_status()
        j = r.json()
        msg = (j.get("message") or {}).get("content")
        if msg:
            return msg
        return j.get("response", "")
    except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
        raise LLMError(f"LLM call failed: {e} [hint: check OLLAMA_URL or increase CHAT timeouts]") from e
    except requests.exceptions.RequestException as e:
        raise LLMError(f"LLM request failed: {e}") from e
    except Exception as e:
        raise LLMError(f"Unexpected error in LLM call: {e}") from e


def __getattr__(name: str) -> str:
    """Dynamically resolve derived attributes such as ``SYSTEM_PROMPT``."""
    if name == "SYSTEM_PROMPT":
        return get_system_prompt()
    raise AttributeError(name)


__all__ = [
    "expand_query",
    "embed_query",
    "normalize_scores_zscore",
    "DenseScoreStore",
    "retrieve",
    "rerank_with_llm",
    "pack_snippets",
    "coverage_ok",
    "ask_llm",
    "tokenize",
    "count_tokens",
    "truncate_to_token_budget",
    "RETRIEVE_PROFILE_LAST",
    "get_system_prompt",
    "SYSTEM_PROMPT",
    "USER_WRAPPER",
    "RERANK_PROMPT",
]
