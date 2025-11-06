# Architecture Vision

## Modularization Roadmap
1. **Retriever module**
   - Extract `retrieve`, `DenseScoreStore`, and query expansion utilities into `clockify_rag.retriever`.
   - Provide a clean interface that accepts query text, embeddings, and BM25 indexes; return ranked chunk IDs plus diagnostics.
2. **Packer module**
   - Move `pack_snippets`, coverage gating, and context budgeting into `clockify_rag.packer` with pluggable token counters.
   - Support alternate packers (e.g., structured answers) via strategy pattern.
3. **Responder module**
   - House `generate_llm_answer`, refusal handling, and citation validation logic.
   - Expose streaming APIs to support SSE/WebSocket frontends.
4. **Service layer**
   - Build a thin orchestration layer (`answer_once`) that composes retriever, packer, and responder components; remove duplicated logic between CLI and package.

## Plugin & Extension Strategy
- Define formal interfaces in `clockify_rag.plugins` for:
  - Custom retrievers (e.g., hybrid + cross-encoder)
  - Rerankers (LLM, gradient-boosted models)
  - Output renderers (Markdown, JSON, structured JSON Schema)
- Provide registration via entry points so third parties can ship add-ons without modifying core code.
- Add metadata contracts (chunk provenance, score explanations) to improve interoperability with dashboards.

## API Design
- Publish a Python client that exposes:
  ```python
  from clockify_rag import RagService

  service = RagService.from_artifacts(pathlib.Path("./kb"))
  answer = service.answer(question, top_k=12, rerank=True)
  ```
- Offer optional FastAPI/Flask service wrapper with JWT auth, rate limiting, and streaming responses.
- Document JSON schemas for requests/responses, including citations and confidence scores.

## Scaling & Performance Strategy
1. **Index build**
   - Parallelize embedding generation via worker pools and persist embeddings in memory-mapped float16 format.
   - Introduce incremental build pipeline (detect changed chunks, update ANN index incrementally).
2. **Retrieval**
   - Cache FAISS indices in shared memory for multi-process deployments.
   - Add adaptive candidate selection (auto-tune `FAISS_CANDIDATE_MULTIPLIER` based on corpus size).
3. **Response generation**
   - Support partial streaming of LLM responses; fall back to batched completion when models do not support streaming.
4. **Operations**
   - Integrate Prometheus metrics (latency histogram, cache hit rate, rerank usage) and expose health checks for build artifacts.

