# Quick Wins (Top 10)

Each item below can be delivered in <30 minutes, carries low risk, and has visible impact.

1. **Fix FAISS branch syntax regression**
   _Impact:_ Restores ANN speedups (~10x faster retrieval).
   _Snippet:_
   ```python
   # clockify_support_cli_final.py, lines 1577-1583
   # Remove the orphaned lines 1581-1582:
   if remaining_idx.size:
       dot_start = time.perf_counter()
       dense_scores_full[remaining_idx] = vecs_n[remaining_idx].dot(qv_n)
       dot_elapsed = time.perf_counter() - dot_start
       dense_computed = int(remaining_idx.size)
   # Delete these two lines:
   #     max(ANN_CANDIDATE_MIN, top_k * FAISS_CANDIDATE_MULTIPLIER)
   # )
   distances = np.asarray(D[0], dtype="float32")
   ```

2. **Initialize dense_scores in HNSW branch**
   _Impact:_ Prevents NameError when FAISS disabled but HNSW available.
   _Snippet:_
   ```python
   # clockify_support_cli_final.py, after line 1595
   elif hnsw:
       _, cand = hnsw.knn_query(qv_n, k=max(ANN_CANDIDATE_MIN, top_k * FAISS_CANDIDATE_MULTIPLIER))
       candidate_idx = cand[0].tolist()
       dense_scores_full = vecs_n.dot(qv_n)
       dense_scores = dense_scores_full[candidate_idx]  # ADD THIS LINE
   ```

3. **Remove duplicate candidate assignments in dense fallback**
   _Impact:_ Avoids redundant dot product (~2x speedup).
   _Snippet:_
   ```python
   # clockify_support_cli_final.py, lines 1596-1604
   # Replace with:
   else:
       dot_start = time.perf_counter()
       dense_scores_full = vecs_n.dot(qv_n)
       dot_elapsed = time.perf_counter() - dot_start
       dense_computed = n_chunks
       dense_scores = dense_scores_full
       candidate_idx = np.arange(n_chunks).tolist()
   ```

4. **Deduplicate FAISS candidate ids**
   _Impact:_ Improves diversity for packing and MMR.
   _Snippet:_
   ```python
   # clockify_support_cli_final.py, before line 1611
   candidate_idx = sorted(set(int(i) for i in candidate_idx if 0 <= i < n_chunks))
   candidate_idx_array = np.array(candidate_idx, dtype=np.int32)
   ```

5. **Provide safe default when hybrid_full falls back**
   _Impact:_ Accurate KPI logs (zero scores mislead filters).
   _Snippet:_
   ```python
   # clockify_support_cli_final.py, line 1649
   # Replace:
   # hybrid_full = np.zeros(len(chunks), dtype="float32")
   # With:
   hybrid_full = np.full(len(chunks), -1.0, dtype="float32")
   for idx, score in zip(candidate_idx, hybrid):
       hybrid_full[idx] = score
   ```

6. **Guard sanitize_question against empty input**
   _Impact:_ Eliminates false negatives when question becomes empty after strip.
   _Snippet:_
   ```python
   # clockify_support_cli_final.py, around line 2174
   def sanitize_question(q: str) -> str:
       q = q.strip()
       if not q:
           raise ValueError("Question cannot be empty. Hint: Provide a meaningful question about Clockify.")
       # ... rest of function
   ```

7. **Default benchmark harness to fake remote**
   _Impact:_ Allows CI to run benchmarks offline without Ollama.
   _Snippet:_
   ```python
   # benchmark.py, line 33
   # Change from:
   # use_real = os.environ.get("BENCHMARK_FAKE_REMOTE") != "1"
   # To:
   use_real = os.environ.get("BENCHMARK_REAL_REMOTE") == "1"
   if not use_real:
       _patch_fake_remote()
   ```

8. **Require auth token in DeepSeek shim**
   _Impact:_ Prevents accidental public exposure of API credentials.
   _Snippet:_
   ```python
   # deepseek_ollama_shim.py, around line 69
   AUTH_TOKEN = os.getenv("SHIM_AUTH_TOKEN")
   if not AUTH_TOKEN:
       raise SystemExit("ERROR: Set SHIM_AUTH_TOKEN environment variable to secure the proxy")
   ```

9. **Add set -euo pipefail to shell scripts**
   _Impact:_ Avoids silent failures in automation pipelines.
   _Snippet:_
   ```bash
   # scripts/acceptance_test.sh, scripts/benchmark.sh, scripts/smoke.sh (line 2)
   #!/bin/bash
   set -euo pipefail  # ADD THIS LINE
   # ... rest of script
   ```

10. **Remove legacy CLI stub**
    _Impact:_ Prevents accidental import of outdated 17-line placeholder module.
    _Snippet:_
    ```bash
    # Terminal command:
    rm clockify_support_cli.py
    # Or replace with single-line redirect:
    echo "# Deprecated: use clockify_support_cli_final" > clockify_support_cli.py
    ```
