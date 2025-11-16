# Deployment Guide

The Clockify RAG stack is designed to run on both Apple Silicon laptops (for local development) and linux/amd64 containers (for production). This guide focuses on deploying the service against the VPN-only Ollama endpoint (`http://10.127.0.192:11434`).

## Supported targets

| Environment | Status | Notes |
|-------------|--------|-------|
| macOS (M1/M2/M3) | ✅ | Use `requirements-m1.txt` or Conda for FAISS. Default `RAG_LLM_CLIENT=mock` keeps workflows offline unless overridden. |
| Linux (x86_64) | ✅ | Preferred production target. Compatible with Docker/Compose. |
| Windows | ⚠️ | Supported via WSL2 only. |

## Baseline deployment (bare metal or VM)

1. **Clone & prepare**
   ```bash
   git clone <repository-url> 1rag
   cd 1rag
   python3 -m venv rag_env && source rag_env/bin/activate
   pip install -r requirements.txt
   cp .env.example .env
   ```

2. **Configure `.env`** – the important entries are:
   ```bash
   RAG_OLLAMA_URL=http://10.127.0.192:11434   # or http://127.0.0.1:11434 locally
   RAG_CHAT_MODEL=qwen2.5:32b
   RAG_EMBED_MODEL=nomic-embed-text:latest
   EMB_BACKEND=ollama                           # use remote embeddings in prod
   RAG_LLM_CLIENT=mock                         # optional, set to `mock` for CI/offline
   ```
   Timeouts (`CHAT_READ_TIMEOUT`, `EMB_READ_TIMEOUT`) and retries (`DEFAULT_RETRIES`) are also configurable for slower VPN links.

3. **Build the artifacts**
   ```bash
   ragctl ingest --input knowledge_full.md
   # or the legacy wrapper:
   python3 clockify_support_cli_final.py build knowledge_full.md
   ```

4. **Run a smoke test**
   ```bash
   # Offline (mock client)
   make smoke

   # Against the remote Ollama host (VPN required)
   RAG_OLLAMA_URL=http://10.127.0.192:11434 \
   SMOKE_CLIENT=ollama make smoke
   ```
   The `make smoke` target now invokes `scripts/smoke_rag.py`, which loads the local index, calls `answer_once`, and reports routing + timing stats. Non-zero exit codes indicate an unhealthy deployment.

5. **Start the API**
   ```bash
   uvicorn clockify_rag.api:app --host 0.0.0.0 --port 8000
   ```
   Use `ragctl doctor --json` and `curl http://<host>:8000/health` to verify readiness.

## Remote Ollama host (VPN)

- The production LLM lives at `http://10.127.0.192:11434` and does **not** require an API key, but is only reachable from the corporate network/VPN.
- Make sure the models are pulled on the server once:
  ```bash
  ollama pull qwen2.5:32b
  ollama pull nomic-embed-text:latest
  ```
- Increase read timeouts when running across the VPN:
  ```bash
  CHAT_READ_TIMEOUT=300
  EMB_READ_TIMEOUT=180
  DEFAULT_RETRIES=4
  ```
- Keep `RAG_LLM_CLIENT=mock` in CI so automated tests never hit the remote host.

## Docker / Compose

### Single container
```bash
docker build -t clockify-rag .

docker run -d \
  --name clockify-rag \
  -p 8000:8000 \
  -e RAG_OLLAMA_URL=http://10.127.0.192:11434 \
  -e RAG_CHAT_MODEL=qwen2.5:32b \
  -e RAG_EMBED_MODEL=nomic-embed-text:latest \
  -e EMB_BACKEND=ollama \
  -v $(pwd)/knowledge_full.md:/app/knowledge_full.md:ro \
  -v $(pwd)/data:/app/data \
  clockify-rag
```

### docker-compose
`docker-compose.yml` already wires the RAG app to an optional Ollama container. Update the `RAG_` env vars as needed, then run:
```bash
docker compose up -d
```
For Apple Silicon hosts building amd64 images, use `docker buildx build --platform linux/amd64`.

## Post-deploy checklist

1. `ragctl doctor --json` – confirms config, index status, and Ollama connectivity.
2. `scripts/smoke_rag.py --question "How do I track time?"` – must return `✅ Smoke test succeeded.`
3. `curl http://localhost:8000/health` – should report `status="healthy"` once the index is loaded.
4. `curl http://localhost:8000/v1/metrics` – verify counters are incrementing when you run queries.
5. Tail `logs/rag_queries.jsonl` (or `$RAG_LOG_FILE`) for structured query logs.

## Platform notes

- **Apple Silicon**: Prefer Conda or the wheels listed in `requirements-m1.txt`. FAISS automatically falls back to FlatIP with a smaller `nlist` for stability.
- **Linux containers**: Ensure `/tmp` and the working directory are writable; ingestion writes `chunks.jsonl`, `vecs_n.npy`, `bm25.json`, `index.meta.json`, etc.

## Troubleshooting

Issue | Fix
----- | ----
`answer_once` refuses due to coverage | Rebuild the index and verify that `chunks.jsonl` is current. Check that `EMB_BACKEND` matches the stored embeddings (see logs at startup).
`LLMUnavailableError` | Confirm VPN connectivity to `RAG_OLLAMA_URL`, increase `CHAT_READ_TIMEOUT`, and inspect `scripts/smoke_rag.py` output for retry hints.
Index build hangs | Remove `.build.lock` if a previous build crashed, then rerun `ragctl ingest --force`.
Docker image cannot reach Ollama | Pass the host via `RAG_OLLAMA_URL=http://host.docker.internal:11434` or expose the VPN route into the container network.

Refer to `docs/RUNBOOK.md` for a deeper operational checklist (alert response, log triage, rebuild workflow).
