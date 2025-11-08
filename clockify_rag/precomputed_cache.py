"""Precomputed query cache for frequently asked questions.

OPTIMIZATION (Analysis Section 9.1 #3): Pre-generate answers for top 100 FAQs.
This module provides a precomputed cache that can store answers for common questions,
achieving 100% cache hit rate for FAQ queries.

Usage:
    # Build precomputed cache from FAQ list
    from clockify_rag.precomputed_cache import build_faq_cache

    faqs = [
        "How do I track time in Clockify?",
        "What are the pricing plans?",
        "Can I use Clockify offline?"
    ]

    build_faq_cache(faqs, chunks, vecs_n, bm, output_path="faq_cache.json")

    # Load and use precomputed cache
    from clockify_rag.precomputed_cache import PrecomputedCache

    cache = PrecomputedCache()
    cache.load("faq_cache.json")

    result = cache.get("How do I track time in Clockify?")
    if result:
        print(result["answer"])
    else:
        # Fall back to normal retrieval
        result = answer_once(question, chunks, vecs_n, bm)
"""

import hashlib
import json
import logging
import os
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class PrecomputedCache:
    """Cache for precomputed answers to frequently asked questions.

    This cache provides O(1) lookup for common questions, achieving instant
    response times for FAQ queries.
    """

    def __init__(self, cache_path: Optional[str] = None):
        """Initialize precomputed cache.

        Args:
            cache_path: Path to cache file (optional, can load later)
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_path = cache_path

        if cache_path and os.path.exists(cache_path):
            self.load(cache_path)

    def _normalize_question(self, question: str) -> str:
        """Normalize question for cache lookup.

        Applies:
        - Lowercase
        - Strip whitespace
        - Remove punctuation

        Args:
            question: User question

        Returns:
            Normalized question string
        """
        import re
        q = question.lower().strip()
        # Remove punctuation except spaces
        q = re.sub(r'[^\w\s]', '', q)
        # Normalize whitespace
        q = ' '.join(q.split())
        return q

    def _hash_question(self, question: str) -> str:
        """Generate hash key for question.

        Args:
            question: Normalized question

        Returns:
            MD5 hash hex string
        """
        return hashlib.md5(question.encode('utf-8')).hexdigest()

    def get(self, question: str, fuzzy: bool = True) -> Optional[Dict[str, Any]]:
        """Get precomputed answer for question.

        Args:
            question: User question
            fuzzy: If True, normalize question before lookup (default: True)

        Returns:
            Cached answer dict or None if not found
        """
        if fuzzy:
            question = self._normalize_question(question)

        key = self._hash_question(question)
        return self.cache.get(key)

    def put(self, question: str, answer_data: Dict[str, Any]) -> None:
        """Store precomputed answer.

        Args:
            question: User question (will be normalized)
            answer_data: Answer dict from answer_once()
        """
        normalized = self._normalize_question(question)
        key = self._hash_question(normalized)

        # Store with metadata
        self.cache[key] = {
            "question_normalized": normalized,
            "question_original": question,
            "answer": answer_data.get("answer"),
            "confidence": answer_data.get("confidence"),
            "refused": answer_data.get("refused"),
            "packed_chunks": answer_data.get("packed_chunks", []),
            "metadata": answer_data.get("metadata", {}),
            "routing": answer_data.get("routing", {}),
        }

    def load(self, cache_path: str) -> None:
        """Load precomputed cache from disk.

        Args:
            cache_path: Path to JSON cache file
        """
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.cache = data.get("cache", {})
                logger.info(f"Loaded precomputed cache: {len(self.cache)} entries from {cache_path}")
        except FileNotFoundError:
            logger.warning(f"Precomputed cache file not found: {cache_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse precomputed cache: {e}")

    def save(self, cache_path: Optional[str] = None) -> None:
        """Save precomputed cache to disk.

        Args:
            cache_path: Path to JSON cache file (uses self.cache_path if not provided)
        """
        if cache_path is None:
            cache_path = self.cache_path

        if cache_path is None:
            raise ValueError("No cache_path specified")

        # Ensure directory exists
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "count": len(self.cache),
            "cache": self.cache
        }

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved precomputed cache: {len(self.cache)} entries to {cache_path}")

    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()

    def size(self) -> int:
        """Get number of cached entries."""
        return len(self.cache)

    def keys(self) -> List[str]:
        """Get list of cached question hashes."""
        return list(self.cache.keys())


def build_faq_cache(
    questions: List[str],
    chunks: List[Dict],
    vecs_n,
    bm: Dict,
    output_path: str = "faq_cache.json",
    **answer_kwargs
) -> PrecomputedCache:
    """Build precomputed cache from list of FAQ questions.

    Args:
        questions: List of frequently asked questions
        chunks: Chunk data
        vecs_n: Normalized embeddings
        bm: BM25 index
        output_path: Where to save cache (default: faq_cache.json)
        **answer_kwargs: Additional arguments for answer_once()

    Returns:
        PrecomputedCache instance with precomputed answers
    """
    from .answer import answer_once

    cache = PrecomputedCache()

    logger.info(f"Building FAQ cache for {len(questions)} questions...")

    for i, question in enumerate(questions, 1):
        logger.info(f"Processing FAQ {i}/{len(questions)}: {question[:60]}...")

        try:
            result = answer_once(question, chunks, vecs_n, bm, **answer_kwargs)
            cache.put(question, result)
        except Exception as e:
            logger.error(f"Failed to process FAQ: {question[:60]}: {e}")

    # Save to disk
    cache.save(output_path)

    logger.info(f"FAQ cache built: {cache.size()} entries saved to {output_path}")
    return cache


def load_faq_list(faq_file: str) -> List[str]:
    """Load FAQ questions from text file (one per line).

    Args:
        faq_file: Path to text file with questions

    Returns:
        List of questions
    """
    questions = []
    with open(faq_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                questions.append(line)

    return questions


# Global precomputed cache instance (lazily loaded)
_PRECOMPUTED_CACHE: Optional[PrecomputedCache] = None


def get_precomputed_cache(cache_path: str = "faq_cache.json") -> PrecomputedCache:
    """Get global precomputed cache instance.

    Args:
        cache_path: Path to cache file

    Returns:
        PrecomputedCache instance
    """
    global _PRECOMPUTED_CACHE

    if _PRECOMPUTED_CACHE is None:
        _PRECOMPUTED_CACHE = PrecomputedCache(cache_path)

    return _PRECOMPUTED_CACHE


__all__ = [
    "PrecomputedCache",
    "build_faq_cache",
    "load_faq_list",
    "get_precomputed_cache",
]
