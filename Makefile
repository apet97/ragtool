.PHONY: help venv install selftest build chat smoke clean dev test eval benchmark benchmark-quick typecheck lint format pre-commit-install pre-commit-run

help:
	@echo "v4.1 Clockify RAG CLI - Make Targets"
	@echo ""
	@echo "  make dev                 - Setup development environment (venv + install + pre-commit)"
	@echo "  make venv                - Create Python virtual environment"
	@echo "  make install             - Install dependencies (requires venv)"
	@echo "  make build               - Build knowledge base (uses local embeddings for speed)"
	@echo "  make selftest            - Run self-test suite"
	@echo "  make chat                - Start interactive chat (REPL)"
	@echo "  make smoke               - Run full smoke test suite"
	@echo "  make test                - Run unit tests with coverage"
	@echo "  make eval                - Run RAG evaluation on ground truth dataset"
	@echo "  make benchmark           - Run performance benchmarks (latency, throughput, memory)"
	@echo "  make benchmark-quick     - Run quick benchmarks (fewer iterations)"
	@echo "  make typecheck           - Run mypy static type checking"
	@echo "  make lint                - Run ruff linter"
	@echo "  make format              - Format code with black"
	@echo "  make pre-commit-install  - Install pre-commit git hooks"
	@echo "  make pre-commit-run      - Run pre-commit hooks on all files"
	@echo "  make clean               - Remove generated artifacts and cache"
	@echo ""
	@echo "Quick start:"
	@echo "  make dev  (then follow on-screen instructions)"
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
	source rag_env/bin/activate && python3 clockify_support_cli_final.py --selftest

chat:
	@echo "Starting interactive chat (REPL)..."
	source rag_env/bin/activate && python3 clockify_support_cli_final.py chat

smoke:
	@echo "Running smoke test suite..."
	bash scripts/smoke.sh

test:
	@echo "Running unit tests with coverage..."
	python3 -m pytest tests/ -v --cov=clockify_support_cli_final --cov-report=term-missing --cov-report=html

eval:
	@echo "Running RAG evaluation on ground truth dataset..."
	python3 eval.py

benchmark:
	@echo "Running performance benchmarks..."
	python3 benchmark.py

benchmark-quick:
	@echo "Running quick benchmarks..."
	python3 benchmark.py --quick

typecheck:
	@echo "Running mypy type checking..."
	python3 -m mypy clockify_support_cli_final.py --config-file pyproject.toml

lint:
	@echo "Running ruff linter..."
	python3 -m ruff check clockify_support_cli_final.py --config pyproject.toml

format:
	@echo "Formatting code with black..."
	python3 -m black clockify_support_cli_final.py --config pyproject.toml

pre-commit-install:
	@echo "Installing pre-commit hooks..."
	python3 -m pre_commit install
	@echo "✅ Pre-commit hooks installed"

pre-commit-run:
	@echo "Running pre-commit hooks on all files..."
	python3 -m pre_commit run --all-files

clean:
	@echo "Cleaning generated artifacts..."
	rm -f chunks.jsonl vecs_n.npy vecs.npy meta.jsonl bm25.json index.meta.json
	rm -f faiss.index hnsw_cosine.bin emb_cache.jsonl
	rm -f .build.lock .shim.pid shim.log build.log smoke.log query.log audit.jsonl
	rm -rf .mypy_cache .pytest_cache htmlcov .ruff_cache
	@echo "✅ Clean complete"

dev: venv install pre-commit-install
	@echo ""
	@echo "✅ Development environment ready!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Activate venv: source rag_env/bin/activate"
	@echo "  2. Build index: make build"
	@echo "  3. Start chat: make chat"
	@echo ""
	@echo "Or run all steps: source rag_env/bin/activate && make build && make chat"
	@echo ""
