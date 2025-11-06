.PHONY: help venv install selftest build chat smoke clean

help:
	@echo "v4.1 Clockify RAG CLI - Make Targets"
	@echo ""
	@echo "  make venv       - Create Python virtual environment"
	@echo "  make install    - Install dependencies (requires venv)"
	@echo "  make build      - Build knowledge base (uses local embeddings for speed)"
	@echo "  make selftest   - Run self-test suite"
	@echo "  make chat       - Start interactive chat (REPL)"
	@echo "  make smoke      - Run full smoke test suite"
	@echo "  make clean      - Remove generated artifacts and cache"
	@echo ""
	@echo "Quick start:"
	@echo "  make venv && make install && make build && make chat"
	@echo ""

venv:
	@echo "Creating virtual environment..."
	python3 -m venv rag_env
	@echo "✅ venv created. Activate: source rag_env/bin/activate"

install:
	@echo "Installing dependencies..."
	@if [ -f requirements.lock ]; then \
		echo "Installing from lockfile..."; \
		source rag_env/bin/activate && pip install -q -r requirements.lock; \
	else \
		echo "Installing from requirements.txt..."; \
		source rag_env/bin/activate && pip install -q -r requirements.txt; \
	fi
	@echo "✅ Dependencies installed"

.PHONY: freeze
freeze:
	@echo "Generating requirements.lock..."
	source rag_env/bin/activate && pip freeze > requirements.lock
	@echo "✅ Lockfile generated"

build:
	@echo "Building knowledge base with local embeddings (faster than Ollama)..."
	source rag_env/bin/activate && EMB_BACKEND=local python3 clockify_support_cli_final.py build knowledge_full.md
	@echo ""
	@echo "Hint: To use Ollama embeddings instead, run: EMB_BACKEND=ollama make build"

selftest:
	@echo "Running self-test suite..."
	source rag_env/bin/activate && python3 clockify_support_cli_final.py selftest

chat:
	@echo "Starting interactive chat (REPL)..."
	source rag_env/bin/activate && python3 clockify_support_cli_final.py chat

smoke:
	@echo "Running smoke test suite..."
	bash scripts/smoke.sh

test:
	@echo "Running unit tests with coverage..."
	python3 -m pytest tests/ -v --cov=clockify_support_cli_final --cov-report=term-missing --cov-report=html

clean:
	@echo "Cleaning generated artifacts..."
	rm -f chunks.jsonl vecs_n.npy vecs.npy meta.jsonl bm25.json index.meta.json
	rm -f faiss.index hnsw_cosine.bin emb_cache.jsonl
	rm -f .build.lock .shim.pid shim.log build.log smoke.log
	@echo "✅ Clean complete"
