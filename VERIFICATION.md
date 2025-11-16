# Environment Verification Guide

Use this checklist whenever you provision a fresh environment (laptop, CI runner, Docker image) to guarantee the Clockify RAG stack is dependency-complete and ready for production use.

## 1. Local Python Environment

1. **Create a virtual environment (Python 3.11)**:
    ```bash
    cd 1rag
    python3 -m venv rag_env
    source rag_env/bin/activate
    pip install -U pip
    ```
2. **Install the package**:
    - Development workflow (tests, lint, doctor):
      ```bash
      pip install -e '.[dev]'
      ```
    - Production/runtime workflow:
      ```bash
      pip install .
      # or use the lockfile when present:
      pip install -r requirements.lock
      ```
3. **Dependency integrity**:
    ```bash
    python -m pip check
    python -m pip list | grep -E 'clockify-rag|torch|faiss|sentence-transformers'
    ```
4. **Makefile gates** – keep `make deps-check` green (pip check + targeted pytest) and use `make test-quick` when you only need the pytest subset:
    ```bash
    make deps-check    # pip check + targeted pytest
    make test-quick    # pytest subset only
    ```
5. **Configuration doctor** – validates Python version, FAISS/BM25 artifacts, and Ollama connectivity:
    ```bash
    ragctl doctor --verbose
    ragctl doctor --json > .verification/doctor.json
    ```
6. **Smoke test retrieval** – defaults to the deterministic mock client so it is safe offline:
    ```bash
    make smoke
    SMOKE_CLIENT=ollama make smoke   # optional, VPN-only
    ```
7. **One-shot verification** – combine `pip check`, the quick pytest subset, and the smoke harness in one command:
    ```bash
    make verify
    SMOKE_CLIENT=ollama make verify   # run the same gate against the real endpoint
    ```

## 2. Docker Image Verification

1. **Build the image**:
    ```bash
    docker build -t clockify-rag .
    ```
2. **Dependency check inside the container**:
    ```bash
    docker run --rm clockify-rag python -m pip check
    ```
3. **Doctor / smoke from the container** (mock client to avoid hitting Ollama):
    ```bash
    docker run --rm \
      -e RAG_LLM_CLIENT=mock \
      -v $(pwd)/chunks.jsonl:/app/chunks.jsonl \
      -v $(pwd)/bm25.json:/app/bm25.json \
      clockify-rag ragctl doctor --json

    docker run --rm \
      -e SMOKE_CLIENT=mock \
      -v $(pwd):/app \
      clockify-rag make smoke
    ```
4. **Runtime command** – by default the container runs `uvicorn clockify_rag.api:app`. Override with `ragctl chat` for REPL workflows:
    ```bash
    docker run --rm -p 8000:8000 clockify-rag
    docker run --rm -it clockify-rag ragctl chat
    ```

## 3. Pre-Deployment Gate

Before promoting a build or cutting a release:

- ✅ `pip check` (inside and outside Docker).
- ✅ `make deps-check` (or `make test-quick` when you only need the pytest subset).
- ✅ `make verify` (runs `pip check`, the quick pytest subset, and `make smoke` in mock mode). Re-run with `SMOKE_CLIENT=ollama make verify` before a VPN-backed deploy.
- ✅ `ragctl doctor --json` reports `index_ready=true` and `ollama.connected=true`.
- ✅ `make smoke` (mock) and `SMOKE_CLIENT=ollama make smoke` if VPN reachability is required.
- ✅ `python -m pip list | grep faiss` to confirm FAISS is correctly installed (install via conda on macOS arm64 before running pip).

Document the command outputs (or save JSON) in your deployment runbook so the next operator can reproduce the verification quickly.
