# Architecture Vision

## Mission
Deliver a modular Clockify RAG platform where the CLI, evaluation suite, and future services reuse the same retrieval core, allowing rapid iteration on chunking, indexing, and prompting without duplicating logic.

## Phased Roadmap
1. **Consolidate core modules (Weeks 1-2)**
   - Extract retrieval helpers (`retrieve`, `pack_snippets`, `answer_once`) into `clockify_rag.retrieval`.
   - Replace inline implementations in `clockify_support_cli_final.py` with imports from `clockify_rag`.
   - Update tests to target the package functions directly, keeping the CLI thin.

2. **Stabilise interfaces (Weeks 3-4)**
   - Introduce configuration dataclasses for build/query settings (e.g., `RetrievalConfig`, `GenerationConfig`).
   - Publish clear hooks for embeddings, rerankers, and context packers through the plugin registry.
   - Document the public API surface and enforce it via type checking.

3. **Service enablement (Weeks 5-6)**
   - Build an HTTP microservice that exposes `/build`, `/query`, and `/metrics` endpoints using the shared retrieval module.
   - Wire structured logging and Prometheus metrics (reuse KPI data) for observability.
   - Provide SDK helpers (Python client) that call into the service or local package interchangeably.

4. **Scalability & resilience (Weeks 7-8)**
   - Add persistent vector store support (FAISS on disk + metadata DB) with migration tooling.
   - Implement background index refresh with atomic swaps to avoid downtime during rebuilds.
   - Introduce distributed query caching (e.g., Redis) with configurable TTL.

## Plugin & Extension Strategy
- **Retrievers:** expose well-defined interface returning scored chunks and metadata. Encourage hybrid strategies (BM25 + dense + reranker) via composition.
- **Rerankers:** allow synchronous or async implementations, enforcing JSON schema for outputs to avoid brittle parsing.
- **Embeddings:** support local (SentenceTransformer) and remote (Ollama, OpenAI) providers behind a common batching API with concurrency guards.
- **Indexes:** define lifecycle hooks (`build`, `load`, `search`, `save`) so alternative ANN backends (HNSW, ScaNN) can be integrated without touching CLI code.

## API Design Principles
- Adopt explicit request/response models (pydantic) for `/query` endpoints with fields: `question`, `top_k`, `pack_top`, `use_rerank`, `context_budget`.
- Include deterministic request IDs and seed handling to ensure reproducible answers during debugging.
- Return structured answers: `{ "answer": str, "citations": [...], "confidence": int, "latency_ms": float }`.
- Provide streaming support via server-sent events for long responses.

## Scaling Strategy
- **Indexing:** run builds in dedicated worker process guarded by file lock; emit progress updates via WebSocket or CLI progress bar.
- **Query Path:** load FAISS indexes into shared read-only memory and fork worker processes to leverage copy-on-write savings.
- **Caching:** tiered approach—local in-process LRU for low-latency hits plus optional distributed cache for multi-instance deployments.
- **Observability:** expose metrics endpoint capturing retrieval latency, ANN hits, reranker usage, refusal counts, cache hit rate.

## Governance & DX
- Enforce lint/type/test gates via CI before merge; add smoke tests covering CLI and service flows.
- Maintain a single authoritative set of docs (README + docs/ directory) with automated checks for outdated files.
- Provide migration guides when altering configuration defaults or plugin interfaces.

