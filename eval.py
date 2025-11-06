#!/usr/bin/env python3
"""
Evaluate RAG system on ground truth dataset.

Computes retrieval metrics:
- MRR (Mean Reciprocal Rank): Position of first relevant result
- Precision@K: Fraction of top K results that are relevant
- NDCG@K: Normalized Discounted Cumulative Gain (position-aware)
"""

import json
import sys
import os
import numpy as np

MRR_THRESHOLD = 0.70
PRECISION_THRESHOLD = 0.60
NDCG_THRESHOLD = 0.65

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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


def evaluate(dataset_path="eval_dataset.jsonl", verbose=False):
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
        print("Create eval_dataset.jsonl with ground truth queries and relevant chunks.")
        sys.exit(1)

    # Check if index is built
    if not os.path.exists("chunks.jsonl") or not os.path.exists("vecs_n.npy"):
        print("Error: Knowledge base not built. Run 'make build' first.")
        sys.exit(1)

    # Import after path setup
    try:
        from clockify_support_cli_final import load_index, retrieve
    except ImportError as e:
        print(f"Error importing RAG functions: {e}")
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

    print(f"Loaded {len(dataset)} evaluation queries")

    # Compute metrics
    mrr_scores = []
    precision_at_5_scores = []
    ndcg_at_10_scores = []

    for i, example in enumerate(dataset):
        query = example["query"]
        relevant_ids = set(example["relevant_chunk_ids"])

        try:
            # Retrieve chunks using RAG system
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
                print(f"  Retrieved:   {retrieved_ids[:5]}")
                print(f"  Relevant:    {list(relevant_ids)}")

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
    print("\n" + "="*70)
    print("RAG EVALUATION RESULTS")
    print("="*70)
    print(f"Dataset size:    {results['dataset_size']}")
    print(f"MRR@10:          {results['mrr_at_10']:.3f} (±{results['mrr_std']:.3f})")
    print(f"Precision@5:     {results['precision_at_5']:.3f} (±{results['precision_std']:.3f})")
    print(f"NDCG@10:         {results['ndcg_at_10']:.3f} (±{results['ndcg_std']:.3f})")
    print("="*70)

    # Interpretation
    print("\nINTERPRETATION:")
    if results['mrr_at_10'] >= MRR_THRESHOLD:
        print(f"✅ MRR@10 ≥ {MRR_THRESHOLD:.2f}: Excellent - first relevant result typically in top 2")
    elif results['mrr_at_10'] >= 0.50:
        print("⚠️  MRR@10 ≥ 0.50: Good - first relevant result typically in top 3-4")
    else:
        print("❌ MRR@10 < 0.50: Needs improvement - relevant results ranked too low")

    if results['precision_at_5'] >= PRECISION_THRESHOLD:
        print(f"✅ Precision@5 ≥ {PRECISION_THRESHOLD:.2f}: Excellent - majority of top 5 are relevant")
    elif results['precision_at_5'] >= 0.40:
        print("⚠️  Precision@5 ≥ 0.40: Good - decent relevance in top results")
    else:
        print("❌ Precision@5 < 0.40: Needs improvement - too many irrelevant results")

    if results['ndcg_at_10'] >= NDCG_THRESHOLD:
        print(f"✅ NDCG@10 ≥ {NDCG_THRESHOLD:.2f}: Excellent - relevant results well-ranked")
    elif results['ndcg_at_10'] >= 0.50:
        print("⚠️  NDCG@10 ≥ 0.50: Good - reasonable ranking quality")
    else:
        print("❌ NDCG@10 < 0.50: Needs improvement - ranking quality suboptimal")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG system on ground truth dataset")
    parser.add_argument("--dataset", default="eval_dataset.jsonl", help="Path to evaluation dataset")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-query results")
    args = parser.parse_args()

    results = evaluate(dataset_path=args.dataset, verbose=args.verbose)

    # Exit with appropriate code based on results
    if (
        results['mrr_at_10'] >= MRR_THRESHOLD
        and results['precision_at_5'] >= PRECISION_THRESHOLD
        and results['ndcg_at_10'] >= NDCG_THRESHOLD
    ):
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Metrics below target
