# Quick Wins

| # | Improvement | Impact | Effort | Snippet |
|---|-------------|--------|--------|---------|
| 1 | Make query expansion deterministic to stabilize caching/logging | Medium | 15 min | ```python
expanded_terms = []
for term, synonyms in expansions.items():
    if re.search(rf"\\b{re.escape(term)}\\b", q_lower):
        for syn in synonyms:
            if syn not in expanded_terms:
                expanded_terms.append(syn)
``` |
| 2 | Avoid rebuilding `retrieved_chunks` twice in `answer_once` | Medium | 20 min | ```python
retrieved_chunks = []
for pack_rank, chunk_id in enumerate(ids):
    info = chunk_scores_by_id.get(chunk_id)
    if info is None:
        continue
    idx = info["index"]
    chunk = chunks[idx]
    if not isinstance(chunk, dict):
        continue
    retrieved_chunks.append({
        "id": chunk_id,
        "pack_rank": pack_rank,
        "dense": info["dense"],
        "bm25": info["bm25"],
        "hybrid": info["hybrid"],
        "mmr_rank": mmr_rank_by_id.get(chunk_id),
        "rerank_score": float(rerank_scores.get(idx, 0.0)),
        "chunk": chunk,
    })
``` |
| 3 | Move `QUERY_LOG_DISABLED` guard to top of `log_query` | Medium | 10 min | ```python
def log_query(...):
    if QUERY_LOG_DISABLED:
        return
    normalized_chunks = []
    for chunk in retrieved_chunks:
        ...
``` |
| 4 | Remove duplicate coverage check in `answer_once` | Medium | 5 min | ```python
coverage_pass = coverage_ok(mmr_selected, dense_scores_all, threshold)
if not coverage_pass:
    ...
chunk_id_to_index = {chunk["id"]: idx for idx, chunk in enumerate(chunks) if isinstance(chunk, dict)}
``` |
| 5 | Honor `LOG_QUERY_INCLUDE_ANSWER` flag when writing query logs | High | 20 min | ```python
entry_answer = answer if LOG_QUERY_INCLUDE_ANSWER else LOG_QUERY_ANSWER_PLACEHOLDER
log_entry = {"answer": entry_answer, ...}
``` |
| 6 | Reuse `dense_scores_full` when ANN is disabled | Medium | 10 min | ```python
dense_scores_full = vecs_n.dot(qv_n)
dense_scores = dense_scores_full
candidate_idx = list(range(len(chunks)))
``` |
| 7 | Create log directory before opening `QUERY_LOG_FILE` | Medium | 10 min | ```python
log_path = pathlib.Path(QUERY_LOG_FILE)
log_path.parent.mkdir(parents=True, exist_ok=True)
with log_path.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(log_entry) + "\n")
``` |
| 8 | Seed FAISS training for reproducible IVF centroids | Medium | 15 min | ```python
rng = np.random.default_rng(DEFAULT_SEED)
train_indices = rng.choice(len(vecs), train_size, replace=False)
``` |
| 9 | Update README status banner to reflect audit blockers | Medium | 10 min | ```markdown
**Status**: ⚠️ Hardening in Progress

> Pending fixes: evaluation script regression, cache invalidation updates.
``` |
|10 | Rename or fix `benchmark_retrieval_bm25` to match behavior | Low | 10 min | ```python
result.name = "retrieve_hybrid"
# or
scores = bm25_scores(question, bm)
``` |
