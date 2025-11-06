# Quick Wins (Top 10)

Each item below can be delivered in <30 minutes, carries low risk, and has visible impact.

1. **Fix FAISS branch syntax regression**  
   _Impact:_ Restores ANN speedups.  
   _Snippet:_
   ```python
   # clockify_support_cli_final.py
   if remaining_idx.size:
       dot_start = time.perf_counter()
       dense_scores_full[remaining_idx] = vecs_n[remaining_idx].dot(qv_n)
       dot_elapsed = time.perf_counter() - dot_start
       dense_computed = int(remaining_idx.size)
   # remove stray max(...) and closing paren
   distances = np.asarray(D[0], dtype="float32")
   ```

2. **Reuse packaged QueryCache**  
   _Impact:_ Eliminates race conditions and duplication.  
   _Snippet:_
   ```python
   # clockify_support_cli_final.py
   from clockify_rag.caching import QueryCache, RateLimiter
   # delete the local QueryCache/RateLimiter class definitions
   CACHE = QueryCache(maxsize=200, ttl_seconds=3600)
   ```

3. **Initialize dense_scores in HNSW branch**  
   _Impact:_ Prevents NameError when FAISS disabled.  
   _Snippet:_
   ```python
   elif hnsw:
       _, cand = hnsw.knn_query(qv_n, k=max(...))
       candidate_idx = cand[0].tolist()
       dense_scores_full = vecs_n.dot(qv_n)
       dense_scores = dense_scores_full[candidate_idx]
   ```

4. **Remove duplicate candidate assignments in dense fallback**  
   _Impact:_ Avoids redundant dot product.  
   _Snippet:_
   ```python
   else:
       dense_scores_full = vecs_n.dot(qv_n)
       dense_scores = dense_scores_full
       candidate_idx = np.arange(n_chunks).tolist()
   ```

5. **Deduplicate FAISS candidate ids**  
   _Impact:_ Improves diversity for packing.  
   _Snippet:_
   ```python
   candidate_idx = sorted(set(int(i) for i in candidate_idx if 0 <= i < n_chunks))
   candidate_idx_array = np.array(candidate_idx, dtype=np.int32)
   ```

6. **Provide safe default when hybrid_full falls back**  
   _Impact:_ Accurate KPI logs.  
   _Snippet:_
   ```python
   else:
       hybrid_full = np.full(len(chunks), -1.0, dtype="float32")
       for idx, score in zip(candidate_idx, hybrid):
           hybrid_full[idx] = score
   ```

7. **Short-circuit cache metadata enrichment**  
   _Impact:_ Stops AttributeError when metadata is None.  
   _Snippet:_
   ```python
   answer, metadata, timestamp = self.cache[key]
   metadata = dict(metadata or {})
   metadata.setdefault("timestamp", timestamp)
   ```

8. **Guard sanitize_question against double spaces from strip**  
   _Impact:_ Eliminates false negatives when question becomes empty.  
   _Snippet:_
   ```python
   q = q.strip()
   if not q:
       raise ValueError("Question cannot be empty. Hint: Provide a meaningful question about Clockify.")
   ```

9. **Default benchmark harness to fake remote**  
   _Impact:_ Allows CI to run benchmarks offline.  
   _Snippet:_
   ```python
   use_real = os.environ.get("BENCHMARK_REAL_REMOTE") == "1"
   if not use_real:
       _patch_fake_remote()
   ```

10. **Require auth token in DeepSeek shim**  
    _Impact:_ Prevents accidental public exposure.  
    _Snippet:_
    ```python
    if not AUTH_TOKEN:
        raise SystemExit("Set SHIM_AUTH_TOKEN to secure the proxy")
    ```
