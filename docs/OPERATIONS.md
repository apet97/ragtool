# Operations Guide

This guide documents how to operate the Clockify RAG stack once it has been installed.  It assumes the repository has been cloned and dependencies installed via `pyproject.toml`/`requirements.txt`.

## Quick Reference

| Task | Command |
|------|---------|
| Dependency health gate | `make deps-check` (pip check + pytest smoke) |
| Verify workstation + config | `ragctl doctor --verbose` or `ragctl doctor --json` |
| Build knowledge base | `make ingest` (alias for `ragctl ingest --input knowledge_full.md`) |
| Rebuild / refresh indexes | `make reindex` or `ragctl ingest --force --input knowledge_full.md` |
| Smoke test end-to-end | `make smoke` *(defaults to the mock LLM client; see below)* |
| Run API locally | `uvicorn clockify_rag.api:app --host 0.0.0.0 --port 8000` |
| Tail structured query logs | `tail -f rag_queries.jsonl` (or `$RAG_LOG_FILE`) |

## Environment Profiles

1. **Local development / laptops**
   - `.env` copied from `.env.example`.
   - `RAG_OLLAMA_URL=http://127.0.0.1:11434` when using a local Ollama daemon.
   - Run `make deps-check` after installing dependencies to ensure `pip check` and the pytest smoke pass.
   - Use `RAG_LLM_CLIENT=mock` to stay offline while iterating on retrieval.
2. **Corporate VPN / remote Ollama**
   - Default `RAG_OLLAMA_URL=http://10.127.0.192:11434`.
   - Increase `CHAT_READ_TIMEOUT`/`EMB_READ_TIMEOUT` and `DEFAULT_RETRIES` for higher latency links.
   - Always run `ragctl doctor --json` and `SMOKE_CLIENT=ollama make smoke` (or `python3 scripts/smoke_rag.py --client ollama`) before exposing the API.
3. **CI / automation**
   - Run `make smoke` (defaults to `--client mock`) to validate artifacts without calling the remote host.
   - Cache `chunks.jsonl`, `vecs_n.npy`, and `bm25.json` between runs to avoid rebuilding for every pipeline.

## Routine Operations

### Building and Rebuilding

```bash
# Initial build (local embeddings for speed)
make ingest

# Force rebuild after editing knowledge_full.md
make reindex
```

Both targets eventually call `ragctl ingest`, which enforces a build lock (`.build.lock`) so only one ingestion runs at a time.  The command validates all derived artifacts before declaring success.

### Running Long-Lived Services

1. Ensure artifacts exist (`ragctl doctor` should report `index_ready: true`).
2. Start the API with uvicorn or gunicorn:
   ```bash
   uvicorn clockify_rag.api:app --host 0.0.0.0 --port 8000
   ```
3. Health checks:
   - `GET /health` – confirms application process status and Ollama availability.
   - `GET /v1/metrics` – emits counters/histograms for SLO tracking.
4. Graceful shutdown: send SIGINT/SIGTERM; the app clears in-memory indexes and flushes caches.

### Updating Models

1. Pull the desired models on the Ollama host (one time):
   ```bash
   ollama pull qwen2.5:32b
   ollama pull nomic-embed-text:latest
   ```
2. Update `.env` (or runtime env vars):
   ```bash
   RAG_CHAT_MODEL=qwen2.5:32b
   RAG_EMBED_MODEL=nomic-embed-text:latest
   ```
3. Restart services so the new configuration is logged at startup (`CONFIG ...` banner).

### Rebuilding the Index After Content Changes

1. Update `knowledge_full.md` (or the directory passed to `ragctl ingest --input`).
2. Execute `make reindex` to delete stale artifacts, rebuild chunks/embeddings/FAISS + BM25, and regenerate metadata (`index.meta.json`).
3. Run `make smoke` (mock) followed by `SMOKE_CLIENT=ollama make smoke` if the remote model is reachable.
4. Redeploy or restart any running API servers to pick up the refreshed artifacts.

## Failure Modes & Debugging

| Symptom | Likely Cause | Mitigation |
|---------|--------------|------------|
| `LLMUnavailableError` or timeouts | VPN down, remote Ollama unreachable, timeouts too low | Check `curl $RAG_OLLAMA_URL/api/tags`, increase `CHAT_READ_TIMEOUT`/`DEFAULT_RETRIES`, ensure `RAG_LLM_CLIENT` is not `mock` by mistake. |
| Coverage failure / constant refusals | Index missing chunks, embeddings built with different backend | Rebuild with `make reindex`, ensure `EMB_BACKEND` used during query matches stored embeddings (see startup log). |
| Slow retrieval or high latency | ANN disabled (`ANN=none`), FAISS index missing | Verify `faiss.index` exists, re-run ingestion, and confirm the log line `faiss_index loaded`. |
| API returns HTTP 500 | Check `logs/` or stderr for stack traces, run `ragctl doctor`, and rerun `scripts/smoke_rag.py` with `--debug`. |
| Query log missing / not updating | `--no-log` flag used or `RAG_LOG_INCLUDE_*` set to 0 | Confirm CLI flags and `.env`.  Log path defaults to `rag_queries.jsonl`. |

When in doubt, enable debug logging temporarily (`python3 clockify_support_cli_final.py --log DEBUG chat`) to inspect retrieval scores and chunk IDs.

## Logging & Metrics

- **Structured query log**: Controlled by `RAG_LOG_FILE` and the redaction toggles (`RAG_LOG_INCLUDE_ANSWER`, `RAG_LOG_INCLUDE_CHUNKS`).  Each entry includes request/response metadata for forensic analysis.
- **Application logs**: Use `clockify_rag.logging_config.setup_logging` (JSON or text) for services; CLI defaults to simple stdout logging with platform + config banners.
- **Metrics**: `clockify_rag.metrics` exposes counters (queries, errors), histograms (latency), and gauges.  `GET /v1/metrics` returns the current snapshot; `export_metrics.py` can dump metrics periodically.

## Smoke & Evaluation Workflows

### Smoke test script

`scripts/smoke_rag.py` now defaults to the mock LLM client so it can run in CI/offline settings safely.  Override with `SMOKE_CLIENT=ollama make smoke` or `python3 scripts/smoke_rag.py --client ollama` to exercise the real endpoint:

```bash
# Offline/CI
make smoke

# Full stack
SMOKE_CLIENT=ollama make smoke
# or call the script directly
python scripts/smoke_rag.py --client ollama --question "How do I track time?"
```

### Evaluation harness

`eval.py --dataset eval_datasets/clockify_v1.jsonl` measures retrieval quality (MRR/NDCG) and supports mocked LLM responses (`RAG_LLM_CLIENT=mock`) for determinism.  Add new Q&A pairs under `eval_datasets/` and document thresholds in `docs/EVALUATION.md`.

## Long-Lived Deployments

- **Systemd**: run uvicorn/gunicorn as a service after preparing a Python virtual env and pointing working directory to the repo.  Ensure the service user owns the index artifacts and `.build.lock`.
- **Docker / Compose**: Provided `docker-compose.yml` includes profiles for bundling Ollama and the API.  Set `RAG_OLLAMA_URL=http://ollama:11434` (compose network) or a VPN-accessible host.
- **Monitoring**: ship `rag_queries.jsonl` and API logs to your centralized logging system, scrape `/v1/metrics`, and configure alerts for high error/refusal counts or missing smoke-test runs.

Document every deployment (hostnames, restart commands, cron jobs rebuilding indexes) in your team wiki so future operators can take over quickly.
