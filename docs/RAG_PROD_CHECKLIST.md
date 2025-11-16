# RAG Production Checklist

Use this checklist before promoting a new build or rolling out changes to the internal Clockify RAG stack.

## 0. Dependencies
- [ ] `pyproject.toml`, Dockerfile, and README all target Python **3.11** (same version in CI and runtime).
- [ ] `pip install -e '.[dev]'` (or `pip install -r requirements.lock`) completes without errors and `python -m pip check` passes.
- [ ] `make deps-check` (pip check + pytest smoke) is green on the deployment host.
- [ ] `make test-quick` (config/API client/answer core tests) is green on the deployment host.
- [ ] `make verify` (pip check + quick pytest subset + `make smoke`) is archived as part of the release notes/runbook.
- [ ] Imports audited so every runtime dependency is declared in `pyproject.toml`/`requirements.txt`; unused packages removed.
- [ ] `VERIFICATION.md` steps (doctor, smoke, Docker pip check) executed and archived in the release notes/runbook.

## 1. Configuration
- [ ] `.env` created from `.env.example` and committed to your secrets store (never to git).
- [ ] `RAG_OLLAMA_URL` points to the correct environment (local loopback vs. `http://10.127.0.192:11434` on VPN).
- [ ] `RAG_CHAT_MODEL` and `RAG_EMBED_MODEL` match the models pulled on the Ollama host.
- [ ] Timeouts/retries tuned for the environment (`CHAT_READ_TIMEOUT`, `EMB_READ_TIMEOUT`, `DEFAULT_RETRIES`).
- [ ] Logging options set appropriately (`RAG_LOG_FILE`, `RAG_LOG_INCLUDE_ANSWER`, `RAG_LOG_INCLUDE_CHUNKS`).

## 2. Environment
- [ ] Virtual environment or container built from `pyproject.toml` / `requirements.txt`.
- [ ] `ragctl doctor --json` returns `index_ready=true` and `ollama.connected=true`.
- [ ] Disk paths for artifacts (chunks, embeddings, indexes) are writable by the service user.
- [ ] Optional proxies configured only when `ALLOW_PROXIES=1`.

## 3. Ingestion & Indexing
- [ ] `make ingest` (or `ragctl ingest`) completed without errors and produced updated artifacts.
- [ ] `index.meta.json` reflects the current corpus checksum and embedding backend.
- [ ] Build lock cleared (`.build.lock` deleted) after ingestion completes.
- [ ] `chunk_title_map.json` regenerated if chunk titles changed (`make regen-artifacts`).

## 4. Retrieval & Prompting
- [ ] Retrieval parameters (`DEFAULT_TOP_K`, `DEFAULT_PACK_TOP`, `DEFAULT_THRESHOLD`, `MMR_LAMBDA`) documented for this deployment.
- [ ] Intent classification enabled/disabled intentionally (`USE_INTENT_CLASSIFICATION`).
- [ ] `make smoke` (mock) succeeds locally; `SMOKE_CLIENT=ollama make smoke` succeeds on VPN (or run `python3 scripts/smoke_rag.py --client ...` directly).
- [ ] Prompt templates reviewed (system/user wrappers) for any policy updates.

## 5. Tests & Evaluation
- [ ] `make test-quick` (or at least `pytest tests/test_api_client.py tests/test_config_module.py`) passes in the targeted environment.
- [ ] `make smoke` completes with status `0` (and double-check with `SMOKE_CLIENT=ollama make smoke` before prod).
- [ ] `eval.py --dataset eval_datasets/clockify_v1.jsonl` meets agreed retrieval thresholds (MRR, NDCG).
- [ ] Deterministic mock client regression (if applicable) captured for CI by setting `RAG_LLM_CLIENT=mock`.

## 6. Operations & Runbook
- [ ] `docs/OPERATIONS.md` referenced/updated with any environment-specific notes.
- [ ] Alerting/log shipping configured (query log path, metrics endpoint, uvicorn logs).
- [ ] Deployment procedure documented (systemd service, Docker compose stack, or manual commands).
- [ ] Rollback procedure tested (restore previous artifacts or redeploy prior tag).

Sign off each section before handing the deployment to another teammate or scheduling a go-live window.
