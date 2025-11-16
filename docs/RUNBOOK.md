# Operations Runbook

Operational checklist for the Clockify RAG service. Use this document when bringing the system online, responding to alerts, or validating fixes.

## 1. Health checks

1. **Config + index sanity**
   ```bash
   ragctl doctor --json
   ```
   - `index.index_ready` must be `true`.
   - `ollama.connected` should be `true` when the VPN is up. If false, switch to `RAG_LLM_CLIENT=mock` until connectivity is restored.

2. **API health**
   ```bash
   curl -s http://localhost:8000/health | jq
   ```
   Status meanings:
   - `healthy` – index loaded and Ollama reachable.
   - `degraded` – index loaded but Ollama unavailable (cached answers still work).
   - `unavailable` – index missing; run `ragctl ingest`.

3. **End-to-end smoke test**
   ```bash
   # Offline CI mode
   make smoke

   # Production mode (VPN → remote Ollama)
   RAG_OLLAMA_URL=http://10.127.0.192:11434 \
   SMOKE_CLIENT=ollama make smoke
   ```
   `scripts/smoke_rag.py` prints routing + timing; exit codes other than 0 indicate failure.

## 2. Common incidents

### LLM host unreachable
- Symptoms: `LLMUnavailableError` in logs, `/health` shows `degraded`.
- Actions:
  1. Confirm VPN path to `RAG_OLLAMA_URL` (`curl $RAG_OLLAMA_URL/api/tags`).
  2. Temporarily serve cached responses by keeping `RAG_LLM_CLIENT=mock`.
  3. Increase `CHAT_READ_TIMEOUT` / `EMB_READ_TIMEOUT` if latency is high.

### Retrieval coverage failures
- Symptoms: API returns refusal with `metadata.coverage_check="failed"`.
- Actions:
  1. Rebuild the index – `ragctl ingest --force`.
  2. Ensure `EMB_BACKEND` matches the stored embeddings (look for dimension mismatch logs).
  3. Validate source markdown freshness.

### Index corruption
- Delete `chunks.jsonl`, `vecs_n.npy`, `bm25.json`, `index.meta.json`, and rerun `ragctl ingest`.
- If `.build.lock` persists, remove it after verifying no other build is running.

## 3. Observability

| Signal | Command |
|--------|---------|
| Query logs | `tail -f ${RAG_LOG_FILE:-rag_queries.jsonl}` (structured JSON). |
| Metrics snapshot | `curl http://localhost:8000/v1/metrics`. |
| Structured LLM errors | Search for `"event":"rag.query.failure"` in the logs directory. |

Each `answer_once` call logs `rag.query.start`/`rag.query.complete` with question hashes, latencies, and routing decisions. Use these to triage regression reports.

## 4. Smoke + regression testing

- `make smoke` – single question through the full stack.
- `python eval.py --dataset eval_datasets/clockify_v1.jsonl` – offline retrieval regression (no network dependency).
- `ragctl query --json "<question>"` – returns the exact payload the API serves (includes `metadata`, `timing`, and `routing`).

## 5. Rebuild / rollback procedure

1. Put the service in maintenance (optional) and stop the API process.
2. `rm -f chunks.jsonl vecs_n.npy bm25.json index.meta.json`.
3. `ragctl ingest --input knowledge_full.md`.
4. `make smoke` (mock), then `SMOKE_CLIENT=ollama make smoke SMOKE_ARGS="--question 'How do I track time?'"` against the production Ollama host.
5. Restart the API (`uvicorn clockify_rag.api:app --port 8000`).
6. Monitor `/v1/metrics` for error counters while traffic ramps up.

Document any deviations or manual fixes in the incident tracker so future updates can be automated.
