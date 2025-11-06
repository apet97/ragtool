#!/usr/bin/env python3
"""Offline evaluation runner for the Clockify RAG system.

The script supports two modes of operation:

1. **Full RAG evaluation** – uses the production ``retrieve`` pipeline if the
   hybrid index (``chunks.jsonl`` + ``vecs_n.npy`` + ``bm25.json``) is present
   and importable. This measures the real retrieval stack (dense + BM25 + MMR).
2. **Lexical fallback** – if the hybrid index is unavailable, the script builds
   an in-memory BM25 index directly from ``knowledge_full.md`` using the same
   chunking heuristics as the main application. This path keeps CI lightweight
   (no heavy Torch/SentenceTransformer dependencies) while still validating the
   knowledge base coverage.

Both paths compute the same metrics:

* MRR@10  – position of the first relevant result
* Precision@5 – fraction of relevant results among the top 5 retrieved chunks
* NDCG@10 – position-aware relevance weighting for the top 10 results

The build fails (exit code 1) when any metric falls below the thresholds
defined in ``SUCCESS_THRESHOLDS``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - handled at runtime
    BM25Okapi = None  # type: ignore[assignment]
MRR_THRESHOLD = 0.70
PRECISION_THRESHOLD = 0.60
NDCG_THRESHOLD = 0.65

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SUCCESS_THRESHOLDS = {
    "mrr_at_10": 0.70,
    "precision_at_5": 0.55,
    "ndcg_at_10": 0.60,
}

TOP_K = 12


def compute_mrr(retrieved_ids, relevant_ids):
    """Mean Reciprocal Rank - measures rank of first relevant result.

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank)
        relevant_ids: Set of ground truth relevant document IDs

    Returns:
        float: 1/rank of first relevant result, or 0 if no relevant results
    """
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def compute_precision_at_k(retrieved_ids, relevant_ids, k=5):
    """Precision@K - fraction of top K results that are relevant.

    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: Set of ground truth relevant document IDs
        k: Number of top results to consider

    Returns:
        float: Precision@K score
    """
    if k == 0:
        return 0.0

    retrieved_k = retrieved_ids[:k]
    hits = len(set(retrieved_k) & set(relevant_ids))
    return hits / k


def compute_ndcg_at_k(retrieved_ids, relevant_ids, k=10):
    """Normalized Discounted Cumulative Gain@K - position-aware metric.

    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: Set of ground truth relevant document IDs
        k: Number of top results to consider

    Returns:
        float: NDCG@K score (0-1)
    """
    # DCG: sum of relevances discounted by log position
    dcg = sum(
        1.0 / np.log2(i + 2) if doc_id in relevant_ids else 0.0
        for i, doc_id in enumerate(retrieved_ids[:k])
    )

    # IDCG: DCG of perfect ranking (all relevant docs first)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))

    return dcg / idcg if idcg > 0 else 0.0


def _normalize_text(value: str) -> str:
    """Lowercase + collapse whitespace for robust key matching."""

    return re.sub(r"\s+", " ", value.strip().lower())


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25 fallback."""

    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return [tok for tok in cleaned.split() if tok]


class LexicalRetriever:
    """BM25-based lexical retriever used when the hybrid index is unavailable."""

    def __init__(self, chunks: list[dict]) -> None:
        if BM25Okapi is None:
            raise RuntimeError(
                "rank-bm25 is required for lexical evaluation. Install rank-bm25"
            )

        corpus_tokens = []
        for chunk in chunks:
            fields = " ".join(
                [
                    chunk.get("title", ""),
                    chunk.get("section", ""),
                    chunk.get("text", ""),
                ]
            )
            corpus_tokens.append(_tokenize(fields))
        self._bm25 = BM25Okapi(corpus_tokens)

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[int]:
        scores = self._bm25.get_scores(_tokenize(query))
        order = np.argsort(scores)[::-1][:top_k]
        return order.tolist()


def _load_chunks() -> list[dict]:
    """Load chunk metadata from disk or rebuild from the knowledge base."""

    if os.path.exists("chunks.jsonl"):
        chunks: list[dict] = []
        with open("chunks.jsonl", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    chunks.append(json.loads(line))
        return chunks

    knowledge_path = os.path.join(os.path.dirname(__file__), "knowledge_full.md")
    if not os.path.exists(knowledge_path):
        raise FileNotFoundError(
            "knowledge_full.md is missing. Run `make build` to generate chunks." \
        )

    try:
        from clockify_rag.chunking import build_chunks
    except ImportError as exc:
        raise RuntimeError(
            "clockify_rag.chunking is required to rebuild chunks. Install the "
            "project dependencies (numpy, requests, nltk)."
        ) from exc

    return build_chunks(knowledge_path)


def _build_chunk_lookup(chunks: list[dict]) -> tuple[dict[str, int], dict[str, set[int]], dict[str, set[int]]]:
    """Create helper mappings for resolving dataset references."""

    id_map: dict[str, int] = {}
    title_section_map: dict[str, set[int]] = defaultdict(set)
    title_map: dict[str, set[int]] = defaultdict(set)

    for idx, chunk in enumerate(chunks):
        cid = chunk.get("id")
        if cid is not None:
            id_map[str(cid)] = idx

        title = _normalize_text(chunk.get("title", ""))
        section = _normalize_text(chunk.get("section", ""))
        title_map[title].add(idx)
        title_section_map[f"{title}|{section}"].add(idx)

    return id_map, title_section_map, title_map


def _resolve_relevant_indices(
    example: dict,
    id_map: dict[str, int],
    title_section_map: dict[str, set[int]],
    title_map: dict[str, set[int]],
) -> set[int]:
    """Resolve dataset annotations into chunk indices."""

    relevant: set[int] = set()

    for cid in example.get("relevant_chunk_ids", []) or []:
        idx = id_map.get(str(cid))
        if idx is not None:
            relevant.add(idx)

    for chunk_ref in example.get("relevant_chunks", []) or []:
        title = _normalize_text(chunk_ref.get("title", ""))
        section = _normalize_text(chunk_ref.get("section", ""))
        key = f"{title}|{section}"
        if section and key in title_section_map:
            relevant.update(title_section_map[key])
            continue
        if title in title_map:
            relevant.update(title_map[title])

    return relevant


def evaluate(dataset_path="eval_datasets/clockify_v1.jsonl", verbose=False):
    """Run evaluation on dataset.

    Args:
        dataset_path: Path to evaluation dataset JSONL file
        verbose: Print per-query results if True

    Returns:
        dict: Evaluation metrics
    """
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Evaluation dataset not found: {dataset_path}")
        sys.exit(1)

    # Load chunk metadata (from built index or source markdown)
    try:
        chunks = _load_chunks()
    except Exception as exc:  # pragma: no cover - fatal setup error
        print(f"Error preparing chunks: {exc}")
        sys.exit(1)

    # Try to load the production hybrid index; fallback to lexical BM25
    rag_available = False
    retrieval_chunks = chunks
    retrieval_fn = None
    lexical_retriever = None

    if os.path.exists("vecs_n.npy") and os.path.exists("bm25.json"):
        try:
            from clockify_support_cli_final import load_index, retrieve

            print("Loading knowledge base index...")
            result = load_index()
            if result:
                retrieval_chunks, vecs, bm, _ = result
                retrieval_fn = lambda q: retrieve(q, retrieval_chunks, vecs, bm, top_k=TOP_K)[0]
                rag_available = True
        except Exception as exc:  # pragma: no cover - fallback path
            print(f"Warning: Hybrid index unavailable ({exc}). Falling back to lexical BM25.")

    if not rag_available:
        print("Building lexical BM25 index for evaluation...")
        try:
            lexical_retriever = LexicalRetriever(retrieval_chunks)
            retrieval_fn = lexical_retriever.retrieve
        except Exception as exc:  # pragma: no cover - fatal fallback error
            print(f"Error building lexical index: {exc}")
            sys.exit(1)

    # Load index
    print("Loading knowledge base index...")
    try:
        result = load_index()
        if result is None:
            print("Error: Failed to load index")
            sys.exit(1)
        chunks, vecs_n, bm, hnsw = result
    except Exception as e:
        print(f"Error loading index: {e}")
        sys.exit(1)

    # Load evaluation dataset
    dataset = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    id_map, title_section_map, title_map = _build_chunk_lookup(retrieval_chunks)

    print(f"Loaded {len(dataset)} evaluation queries")

    # Compute metrics
    mrr_scores = []
    precision_at_5_scores = []
    ndcg_at_10_scores = []

    skipped = 0
    for i, example in enumerate(dataset):
        query = example["query"]
        relevant_ids = _resolve_relevant_indices(example, id_map, title_section_map, title_map)
        if not relevant_ids:
            print(f"Warning: Skipping query {i+1} ('{query}') - no matching relevant chunks found.")
            skipped += 1
            continue

        try:
            # Retrieve chunks using RAG system
            retrieved_ids = list(retrieval_fn(query)) if retrieval_fn else []
            selected, _ = retrieve(query, chunks, vecs_n, bm, top_k=12, hnsw=hnsw)
            retrieved_ids = list(selected)

            # Compute metrics
            mrr = compute_mrr(retrieved_ids, relevant_ids)
            precision_at_5 = compute_precision_at_k(retrieved_ids, relevant_ids, k=5)
            ndcg_at_10 = compute_ndcg_at_k(retrieved_ids, relevant_ids, k=10)

            mrr_scores.append(mrr)
            precision_at_5_scores.append(precision_at_5)
            ndcg_at_10_scores.append(ndcg_at_10)

            if verbose:
                print(f"\nQuery {i+1}: {query}")
                print(f"  MRR:         {mrr:.3f}")
                print(f"  Precision@5: {precision_at_5:.3f}")
                print(f"  NDCG@10:     {ndcg_at_10:.3f}")
                print(f"  Retrieved idx: {retrieved_ids[:5]}")
                print(f"  Relevant idx:  {sorted(relevant_ids)}")

        except Exception as e:
            print(f"Error evaluating query '{query}': {e}")
            continue

    # Compute aggregate metrics
    results = {
        "dataset_size": len(dataset),
        "mrr_at_10": float(np.mean(mrr_scores)),
        "precision_at_5": float(np.mean(precision_at_5_scores)),
        "ndcg_at_10": float(np.mean(ndcg_at_10_scores)),
        "mrr_std": float(np.std(mrr_scores)),
        "precision_std": float(np.std(precision_at_5_scores)),
        "ndcg_std": float(np.std(ndcg_at_10_scores))
    }

    # Print results
    processed = len(mrr_scores)
    if processed == 0:
        print("Error: No evaluation queries were processed. Check dataset annotations.")
        sys.exit(1)

    print("\n" + "="*70)
    mode = "Hybrid RAG" if rag_available else "Lexical BM25"
    print(f"RAG EVALUATION RESULTS ({mode})")
    print("="*70)
    print(f"Dataset size:    {results['dataset_size']}")
    if skipped:
        print(f"Skipped queries: {skipped} (missing relevance annotations)")
    print(f"MRR@10:          {results['mrr_at_10']:.3f} (±{results['mrr_std']:.3f})")
    print(f"Precision@5:     {results['precision_at_5']:.3f} (±{results['precision_std']:.3f})")
    print(f"NDCG@10:         {results['ndcg_at_10']:.3f} (±{results['ndcg_std']:.3f})")
    print("="*70)

    # Interpretation
    print("\nINTERPRETATION:")
    if results['mrr_at_10'] >= SUCCESS_THRESHOLDS['mrr_at_10']:
        print(
            f"✅ MRR@10 ≥ {SUCCESS_THRESHOLDS['mrr_at_10']:.2f}: "
            "Excellent - first relevant result typically in top 2"
        )
    if results['mrr_at_10'] >= MRR_THRESHOLD:
        print(f"✅ MRR@10 ≥ {MRR_THRESHOLD:.2f}: Excellent - first relevant result typically in top 2")
    elif results['mrr_at_10'] >= 0.50:
        print("⚠️  MRR@10 ≥ 0.50: Good - first relevant result typically in top 3-4")
    else:
        print("❌ MRR@10 < 0.50: Needs improvement - relevant results ranked too low")

    if results['precision_at_5'] >= SUCCESS_THRESHOLDS['precision_at_5']:
        print(
            f"✅ Precision@5 ≥ {SUCCESS_THRESHOLDS['precision_at_5']:.2f}: "
            "Excellent - majority of top 5 are relevant"
        )
    if results['precision_at_5'] >= PRECISION_THRESHOLD:
        print(f"✅ Precision@5 ≥ {PRECISION_THRESHOLD:.2f}: Excellent - majority of top 5 are relevant")
    elif results['precision_at_5'] >= 0.40:
        print("⚠️  Precision@5 ≥ 0.40: Good - decent relevance in top results")
    else:
        print("❌ Precision@5 < 0.40: Needs improvement - too many irrelevant results")

    if results['ndcg_at_10'] >= SUCCESS_THRESHOLDS['ndcg_at_10']:
        print(
            f"✅ NDCG@10 ≥ {SUCCESS_THRESHOLDS['ndcg_at_10']:.2f}: "
            "Excellent - relevant results well-ranked"
        )
    if results['ndcg_at_10'] >= NDCG_THRESHOLD:
        print(f"✅ NDCG@10 ≥ {NDCG_THRESHOLD:.2f}: Excellent - relevant results well-ranked")
    elif results['ndcg_at_10'] >= 0.50:
        print("⚠️  NDCG@10 ≥ 0.50: Good - reasonable ranking quality")
    else:
        print("❌ NDCG@10 < 0.50: Needs improvement - ranking quality suboptimal")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG system on ground truth dataset")
    parser.add_argument(
        "--dataset",
        default="eval_datasets/clockify_v1.jsonl",
        help="Path to evaluation dataset",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-query results")
    parser.add_argument(
        "--min-mrr",
        type=float,
        default=SUCCESS_THRESHOLDS["mrr_at_10"],
        help="Minimum acceptable MRR@10 (default: %(default).2f)",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=SUCCESS_THRESHOLDS["precision_at_5"],
        help="Minimum acceptable Precision@5 (default: %(default).2f)",
    )
    parser.add_argument(
        "--min-ndcg",
        type=float,
        default=SUCCESS_THRESHOLDS["ndcg_at_10"],
        help="Minimum acceptable NDCG@10 (default: %(default).2f)",
    )
    args = parser.parse_args()

    results = evaluate(dataset_path=args.dataset, verbose=args.verbose)

    thresholds = {
        "mrr_at_10": args.min_mrr,
        "precision_at_5": args.min_precision,
        "ndcg_at_10": args.min_ndcg,
    }

    success = all(results[key] >= threshold for key, threshold in thresholds.items())
    if not success:
        print(
            "\nEvaluation thresholds not met:" +
            " ".join(
                f"{key}={results[key]:.3f} < {threshold:.2f}"
                for key, threshold in thresholds.items()
                if results[key] < threshold
            )
        )
        sys.exit(1)

    sys.exit(0)
    # Exit with appropriate code based on results
    if (
        results['mrr_at_10'] >= MRR_THRESHOLD
        and results['precision_at_5'] >= PRECISION_THRESHOLD
        and results['ndcg_at_10'] >= NDCG_THRESHOLD
    ):
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Metrics below target
