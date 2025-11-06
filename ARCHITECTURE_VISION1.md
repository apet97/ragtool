# Architecture Vision: Clockify RAG System Roadmap

**Document Version**: 1.0
**Date**: 2025-11-06
**Timeframe**: 3-6 months
**Target**: Production-grade, scalable, multi-tenant RAG platform

---

## Executive Summary

This document outlines the architectural evolution of the Clockify RAG system from its current state (v5.0 - modular with plugin architecture) to a production-grade, scalable platform supporting:
- **Multi-tenancy**: Multiple knowledge bases per tenant
- **High concurrency**: 100+ concurrent queries per second
- **Continuous evaluation**: Automated quality monitoring and A/B testing
- **Extensibility**: Plugin ecosystem for custom retrievers, rerankers, and embeddings
- **Observability**: Full metrics, tracing, and monitoring

---

## Current State (v5.0)

### Strengths
✅ Modular package structure (`clockify_rag/`)
✅ Plugin architecture (interfaces + registry)
✅ Hybrid retrieval (BM25 + dense + MMR + FAISS ANN)
✅ Comprehensive caching (embeddings + queries)
✅ Production hardening (atomic writes, locks, sanitization)

### Limitations
⚠️ Single knowledge base per deployment
⚠️ Not thread-safe (global state)
⚠️ No evaluation framework
⚠️ CLI-only interface (no REST API)
⚠️ Limited observability (basic logging)
⚠️ No distributed indexing

---

## Target State (v6.0+)

### High-Level Goals

1. **Multi-Tenancy**: Support 100+ tenants with isolated indexes
2. **High Concurrency**: 100+ QPS with <100ms p95 latency
3. **Continuous Evaluation**: Automated metric tracking, A/B testing
4. **REST API**: HTTP/JSON API for programmatic access
5. **Distributed Indexing**: Horizontally scalable index building
6. **Full Observability**: Prometheus metrics, OpenTelemetry tracing, structured logs

---

## Phase 1: Modularization & Thread Safety (Month 1)

### Objectives
- Complete migration from monolithic file to modular package
- Fix thread safety issues
- Add comprehensive test coverage

### Architecture Changes

#### 1.1 Deprecate Monolithic File
**Timeline**: Week 1-2
**Effort**: Medium

```
# Current structure
clockify_support_cli_final.py (2,857 lines, all-in-one)
clockify_rag/ (modular package, partial implementation)

# Target structure
clockify_rag/
  ├── __init__.py
  ├── config.py
  ├── exceptions.py
  ├── utils.py
  ├── http_utils.py
  ├── chunking.py
  ├── embedding.py
  ├── indexing.py
  ├── retrieval.py      # NEW: Hybrid retrieval logic
  ├── reranking.py      # NEW: MMR, LLM reranking
  ├── llm.py            # NEW: LLM interaction
  ├── caching.py
  ├── cli.py            # NEW: CLI/REPL interface
  └── plugins/
      ├── __init__.py
      ├── interfaces.py
      ├── registry.py
      └── builtin/      # NEW: Built-in plugins
          ├── hybrid_retriever.py
          ├── cross_encoder_reranker.py
          └── faiss_index.py
```

**Migration Steps**:
1. Move MMR diversification to `retrieval.py`
2. Move LLM interaction to `llm.py`
3. Move CLI/REPL to `cli.py`
4. Update all imports in tests, benchmarks, eval
5. Add deprecation warning to monolithic file
6. Delete monolithic file after 1-month deprecation period

#### 1.2 Thread Safety Implementation
**Timeline**: Week 2
**Effort**: Low-Medium

**Add locks to shared state**:

```python
# clockify_rag/caching.py
import threading

class QueryCache:
    def __init__(self, maxsize=100, ttl_seconds=3600):
        # ... existing fields ...
        self._lock = threading.RLock()  # Reentrant lock

    def get(self, question: str):
        with self._lock:
            # ... existing logic ...

    def put(self, question: str, answer: str, metadata: dict):
        with self._lock:
            # ... existing logic ...

# clockify_rag/indexing.py
_FAISS_INDEX = None
_FAISS_LOCK = threading.Lock()

def _load_faiss_index_once():
    global _FAISS_INDEX
    if _FAISS_INDEX is not None:
        return _FAISS_INDEX
    with _FAISS_LOCK:
        # Double-checked locking pattern
        if _FAISS_INDEX is not None:
            return _FAISS_INDEX
        _FAISS_INDEX = load_faiss_index(FILES["faiss_index"])
        return _FAISS_INDEX
```

**Testing**:
```python
# tests/test_thread_safety.py
import threading
import pytest

def test_query_cache_thread_safe():
    """Verify cache works correctly with concurrent access."""
    cache = QueryCache()
    results = []

    def worker(question, answer):
        cache.put(question, answer, {})
        result = cache.get(question)
        results.append(result)

    threads = [threading.Thread(target=worker, args=(f"q{i}", f"a{i}")) for i in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 100
    assert all(r is not None for r in results)
```

#### 1.3 Test Coverage Expansion
**Timeline**: Week 3-4
**Effort**: Medium

**Add missing tests**:
```
tests/
  ├── test_retrieval.py          # NEW: Hybrid retrieval tests
  ├── test_mmr.py                # NEW: MMR diversification tests
  ├── test_reranking.py          # NEW: Reranking tests
  ├── test_llm.py                # NEW: LLM interaction tests
  ├── test_integration.py        # NEW: End-to-end tests
  ├── test_thread_safety.py      # NEW: Concurrency tests
  ├── test_faiss.py              # NEW: FAISS index tests
  └── conftest.py                # NEW: Shared fixtures
```

**Target Coverage**: 80% (from current ~20%)

---

## Phase 2: Evaluation Framework & Monitoring (Month 2)

### Objectives
- Create ground truth evaluation dataset
- Implement automated metric tracking
- Add observability infrastructure

### Architecture Changes

#### 2.1 Evaluation Framework
**Timeline**: Week 1-2
**Effort**: Medium

```
clockify_rag/
  └── evaluation/                # NEW: Evaluation module
      ├── __init__.py
      ├── metrics.py             # MRR, NDCG, P@K, answer quality
      ├── dataset.py             # Ground truth dataset management
      ├── evaluator.py           # Run evaluation on dataset
      └── calibration.py         # Confidence calibration

# Ground truth dataset format
eval_datasets/
  ├── clockify_v1.jsonl          # 50-100 examples
  ├── clockify_adversarial.jsonl # Edge cases
  └── clockify_multilingual.jsonl # Non-English queries

# Example entry
{
  "query": "How do I track time in Clockify?",
  "answer": "Click the timer button in the top right corner...",
  "relevant_chunk_ids": ["uuid1", "uuid2", "uuid3"],
  "difficulty": "easy",
  "tags": ["time-tracking", "basic"],
  "language": "en"
}
```

**Automated Evaluation**:
```python
# clockify_rag/evaluation/evaluator.py
class RAGEvaluator:
    """Evaluate RAG system on ground truth dataset."""

    def __init__(self, dataset_path: str):
        self.dataset = load_dataset(dataset_path)

    def evaluate(self, rag_system) -> dict:
        """Run evaluation and return metrics."""
        results = []
        for example in self.dataset:
            retrieved = rag_system.retrieve(example["query"])
            answer = rag_system.answer(example["query"])
            results.append({
                "mrr": compute_mrr(retrieved, example["relevant_chunk_ids"]),
                "ndcg_10": compute_ndcg_at_k(retrieved, example["relevant_chunk_ids"], k=10),
                "precision_5": compute_precision_at_k(retrieved, example["relevant_chunk_ids"], k=5),
                "answer_quality": score_answer(answer, example["answer"]),
            })
        return aggregate_metrics(results)

# CLI integration
# python3 -m clockify_rag.evaluation.evaluator \
#   --dataset eval_datasets/clockify_v1.jsonl \
#   --output eval_results.json
```

#### 2.2 Observability Infrastructure
**Timeline**: Week 3-4
**Effort**: Medium

**Add Prometheus metrics**:

```python
# clockify_rag/observability/
  ├── __init__.py
  ├── metrics.py                 # Prometheus metrics
  ├── tracing.py                 # OpenTelemetry tracing
  └── logging.py                 # Structured logging

# clockify_rag/observability/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Query metrics
query_counter = Counter('rag_queries_total', 'Total queries', ['status'])
query_latency = Histogram('rag_query_latency_seconds', 'Query latency')
cache_hit_counter = Counter('rag_cache_hits_total', 'Cache hits')

# Retrieval metrics
retrieval_latency = Histogram('rag_retrieval_latency_seconds', 'Retrieval latency')
chunks_retrieved = Histogram('rag_chunks_retrieved', 'Chunks retrieved per query')

# Quality metrics (if ground truth available)
mrr_gauge = Gauge('rag_mrr', 'Mean Reciprocal Rank')
ndcg_gauge = Gauge('rag_ndcg_10', 'NDCG@10')
precision_gauge = Gauge('rag_precision_5', 'Precision@5')

# Export endpoint
# GET /metrics → Prometheus format
```

**Add OpenTelemetry tracing**:

```python
# clockify_rag/observability/tracing.py
from opentelemetry import trace
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# Auto-instrument HTTP requests
RequestsInstrumentor().instrument()

tracer = trace.get_tracer(__name__)

# In retrieval pipeline
@tracer.start_as_current_span("hybrid_retrieval")
def retrieve(question: str, ...):
    with tracer.start_as_current_span("bm25_scoring"):
        bm_scores = bm25_scores(question, bm)
    with tracer.start_as_current_span("dense_scoring"):
        dense_scores = vecs_n.dot(qv_n)
    with tracer.start_as_current_span("mmr_diversification"):
        selected = apply_mmr(...)
    return selected
```

**Structured logging**:

```python
# clockify_rag/observability/logging.py
import structlog

logger = structlog.get_logger()

# In query handler
logger.info("query_received",
    query_length=len(question),
    cache_hit=False,
    user_id=user_id)

# In retrieval
logger.info("retrieval_complete",
    query_id=query_id,
    top_k=top_k,
    retrieved=len(selected),
    latency_ms=latency)
```

---

## Phase 3: REST API & Multi-Tenancy (Month 3)

### Objectives
- Add REST API for programmatic access
- Support multiple knowledge bases (multi-tenancy)
- Implement tenant isolation

### Architecture Changes

#### 3.1 REST API Design
**Timeline**: Week 1-2
**Effort**: Medium

```python
# clockify_rag/api/
  ├── __init__.py
  ├── app.py                     # FastAPI application
  ├── routes/
  │   ├── query.py               # POST /query
  │   ├── index.py               # POST /index/build, GET /index/status
  │   └── health.py              # GET /health, GET /metrics
  ├── models.py                  # Pydantic request/response models
  └── middleware.py              # Auth, rate limiting, CORS

# clockify_rag/api/app.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Clockify RAG API", version="6.0")

# Middleware
app.add_middleware(CORSMiddleware, allow_origins=["*"])
Instrumentator().instrument(app).expose(app)

# Routes
@app.post("/query")
async def query_endpoint(
    request: QueryRequest,
    tenant_id: str = Depends(get_tenant_id),
    rate_limit: None = Depends(check_rate_limit)
) -> QueryResponse:
    """Query RAG system."""
    rag = get_rag_instance(tenant_id)
    answer, metadata = rag.answer(request.question)
    return QueryResponse(answer=answer, metadata=metadata)

@app.post("/index/build")
async def build_index(
    request: BuildRequest,
    tenant_id: str = Depends(get_tenant_id),
    auth: None = Depends(require_admin)
) -> BuildResponse:
    """Build knowledge base index."""
    task_id = enqueue_build_task(tenant_id, request.kb_path)
    return BuildResponse(task_id=task_id, status="queued")

# Run server
# uvicorn clockify_rag.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

**API Spec**:

```yaml
# openapi.yaml
openapi: 3.0.0
info:
  title: Clockify RAG API
  version: 6.0.0

paths:
  /query:
    post:
      summary: Query RAG system
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                question:
                  type: string
                  maxLength: 2000
                top_k:
                  type: integer
                  default: 12
                threshold:
                  type: float
                  default: 0.30
      responses:
        200:
          content:
            application/json:
              schema:
                type: object
                properties:
                  answer:
                    type: string
                  confidence:
                    type: number
                  citations:
                    type: array
                    items:
                      type: string
                  metadata:
                    type: object

  /index/build:
    post:
      summary: Build knowledge base
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                kb_url:
                  type: string
                  format: uri
      responses:
        202:
          content:
            application/json:
              schema:
                type: object
                properties:
                  task_id:
                    type: string
                  status:
                    type: string

  /health:
    get:
      summary: Health check
      responses:
        200:
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                  version:
                    type: string
```

#### 3.2 Multi-Tenancy Implementation
**Timeline**: Week 3-4
**Effort**: High

**Tenant Isolation**:

```python
# clockify_rag/tenancy/
  ├── __init__.py
  ├── manager.py                 # Tenant manager
  ├── storage.py                 # Per-tenant file storage
  └── cache.py                   # Per-tenant cache isolation

# clockify_rag/tenancy/manager.py
class TenantManager:
    """Manage multiple RAG instances per tenant."""

    def __init__(self, storage_root: str):
        self.storage_root = storage_root
        self.tenants = {}  # {tenant_id: RAGInstance}
        self._lock = threading.RLock()

    def get_or_create(self, tenant_id: str) -> 'RAGInstance':
        """Get existing tenant or create new one."""
        with self._lock:
            if tenant_id not in self.tenants:
                tenant_dir = os.path.join(self.storage_root, tenant_id)
                os.makedirs(tenant_dir, exist_ok=True)
                self.tenants[tenant_id] = RAGInstance(
                    storage_dir=tenant_dir,
                    cache=QueryCache(maxsize=100),
                    rate_limiter=RateLimiter(max_requests=100)
                )
            return self.tenants[tenant_id]

# Per-tenant file structure
storage/
  ├── tenant_001/
  │   ├── chunks.jsonl
  │   ├── vecs_n.npy
  │   ├── bm25.json
  │   ├── faiss.index
  │   └── index.meta.json
  ├── tenant_002/
  │   └── ...
  └── tenant_003/
      └── ...
```

**Resource Management**:

```python
# Lazy loading: Load index only when first query received
# Eviction: LRU eviction after 1 hour idle time

class RAGInstance:
    """Per-tenant RAG instance with lazy loading."""

    def __init__(self, storage_dir: str, cache: QueryCache, rate_limiter: RateLimiter):
        self.storage_dir = storage_dir
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.index = None  # Lazy-loaded
        self.last_access = time.time()

    def query(self, question: str) -> tuple:
        """Query with lazy index loading."""
        if self.index is None:
            self.index = load_index(self.storage_dir)
        self.last_access = time.time()
        return answer_once(question, *self.index)

# Background eviction task
def evict_idle_tenants(manager: TenantManager, idle_threshold_sec: int = 3600):
    """Evict tenants idle for > 1 hour."""
    now = time.time()
    for tenant_id, instance in list(manager.tenants.items()):
        if now - instance.last_access > idle_threshold_sec:
            del manager.tenants[tenant_id]
            logger.info(f"Evicted tenant {tenant_id} after {idle_threshold_sec}s idle")
```

---

## Phase 4: Distributed Indexing & Scalability (Month 4)

### Objectives
- Horizontally scale index building (parallel workers)
- Support large knowledge bases (100k+ documents)
- Implement distributed retrieval (sharding)

### Architecture Changes

#### 4.1 Distributed Index Building
**Timeline**: Week 1-2
**Effort**: High

**Use Celery for async task queue**:

```python
# clockify_rag/tasks/
  ├── __init__.py
  ├── celery_app.py              # Celery configuration
  ├── build_tasks.py             # Async build tasks
  └── workers.py                 # Worker pool management

# clockify_rag/tasks/celery_app.py
from celery import Celery

app = Celery('clockify_rag',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0')

app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
)

# clockify_rag/tasks/build_tasks.py
from .celery_app import app

@app.task(bind=True, max_retries=3)
def build_index_async(self, tenant_id: str, kb_path: str):
    """Async index building task."""
    try:
        # Update task status
        self.update_state(state='PROGRESS', meta={'progress': 0.0})

        # Step 1: Chunking (25%)
        chunks = build_chunks(kb_path)
        self.update_state(state='PROGRESS', meta={'progress': 0.25})

        # Step 2: Embedding (50%)
        vecs = embed_texts([c["text"] for c in chunks])
        self.update_state(state='PROGRESS', meta={'progress': 0.50})

        # Step 3: BM25 (75%)
        bm = build_bm25(chunks)
        self.update_state(state='PROGRESS', meta={'progress': 0.75})

        # Step 4: FAISS (100%)
        faiss_index = build_faiss_index(vecs)
        self.update_state(state='PROGRESS', meta={'progress': 1.0})

        return {'status': 'completed', 'chunks': len(chunks)}
    except Exception as e:
        self.retry(countdown=60, exc=e)

# API integration
@app.post("/index/build")
async def build_index(request: BuildRequest, tenant_id: str = Depends(get_tenant_id)):
    task = build_index_async.delay(tenant_id, request.kb_path)
    return {"task_id": task.id, "status": "queued"}

@app.get("/index/status/{task_id}")
async def get_build_status(task_id: str):
    task = build_index_async.AsyncResult(task_id)
    return {"task_id": task_id, "status": task.state, "meta": task.info}
```

**Parallel Chunking & Embedding**:

```python
# Use multiprocessing for CPU-bound tasks
from multiprocessing import Pool

def build_chunks_parallel(md_path: str, num_workers: int = 4) -> list:
    """Parallel chunking using multiprocessing."""
    # Split MD into sections
    sections = split_into_sections(md_path)

    # Process sections in parallel
    with Pool(processes=num_workers) as pool:
        chunk_batches = pool.map(process_section, sections)

    # Flatten batches
    return [chunk for batch in chunk_batches for chunk in batch]

def embed_texts_parallel(texts: list, batch_size: int = 32, num_workers: int = 4) -> np.ndarray:
    """Parallel embedding using multiple workers."""
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]

    with Pool(processes=num_workers) as pool:
        vec_batches = pool.map(embed_local_batch, batches)

    return np.vstack(vec_batches)
```

#### 4.2 Index Sharding for Large KBs
**Timeline**: Week 3-4
**Effort**: High

**Shard by document collection**:

```python
# For 100k+ documents, shard index into multiple sub-indexes

# clockify_rag/sharding/
  ├── __init__.py
  ├── sharded_index.py           # Sharded index manager
  └── router.py                  # Query routing

# Storage structure
tenant_001/
  ├── shard_0/
  │   ├── chunks.jsonl           # 10k chunks
  │   ├── vecs_n.npy
  │   ├── bm25.json
  │   └── faiss.index
  ├── shard_1/
  │   └── ...                    # 10k chunks
  ├── ...
  └── shard_9/
      └── ...                    # 10k chunks

# Query routing: Retrieve from all shards, merge results
class ShardedRetriever:
    """Retrieve from multiple shards and merge."""

    def __init__(self, shard_dirs: list):
        self.shards = [load_index(d) for d in shard_dirs]

    def retrieve(self, question: str, top_k: int = 12) -> list:
        """Retrieve from all shards, merge by score."""
        all_results = []
        for shard in self.shards:
            results = shard.retrieve(question, top_k=top_k)
            all_results.extend(results)

        # Merge and re-rank
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]
```

---

## Phase 5: Advanced RAG Features (Month 5)

### Objectives
- Multi-hop query support (decomposition)
- Cross-encoder reranking
- Answer validation and citation checking
- Query reformulation

### Architecture Changes

#### 5.1 Multi-Hop Query Decomposition
**Timeline**: Week 1-2
**Effort**: Medium

```python
# clockify_rag/query/
  ├── __init__.py
  ├── decomposer.py              # Query decomposition
  ├── reformulator.py            # Query reformulation
  └── expander.py                # Query expansion (existing)

# clockify_rag/query/decomposer.py
class QueryDecomposer:
    """Decompose complex queries into sub-queries."""

    def decompose(self, question: str) -> list[str]:
        """Break complex question into 2-3 sub-queries."""
        # Use LLM to decompose question
        prompt = f"""Break this complex question into 2-3 simpler sub-questions:

Question: {question}

Sub-questions (one per line):"""

        response = llm_call(prompt, max_tokens=200)
        sub_queries = [q.strip() for q in response.split('\n') if q.strip()]
        return sub_queries[:3]  # Limit to top 3

# Usage in retrieval pipeline
def retrieve_multi_hop(question: str, chunks, vecs_n, bm, top_k: int = 12):
    """Retrieve with multi-hop decomposition."""
    # Check if question is complex
    if is_complex_question(question):
        sub_queries = decomposer.decompose(question)
        all_results = []
        for sq in sub_queries:
            results = retrieve(sq, chunks, vecs_n, bm, top_k=top_k // len(sub_queries))
            all_results.extend(results)

        # Deduplicate and merge
        unique_results = deduplicate_by_id(all_results)
        return unique_results[:top_k]
    else:
        return retrieve(question, chunks, vecs_n, bm, top_k=top_k)
```

#### 5.2 Cross-Encoder Reranking
**Timeline**: Week 2
**Effort**: Low

```python
# clockify_rag/reranking/
  ├── __init__.py
  ├── cross_encoder.py           # Cross-encoder reranking
  ├── llm_reranker.py            # LLM reranking (existing)
  └── mmr.py                     # MMR diversification (existing)

# clockify_rag/reranking/cross_encoder.py
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    """Rerank using cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, question: str, chunks: list, top_k: int = 6) -> list:
        """Rerank chunks using cross-encoder."""
        # Prepare pairs: (question, chunk_text)
        pairs = [(question, c["text"]) for c in chunks]

        # Predict scores
        scores = self.model.predict(pairs)

        # Sort by score
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [c for c, s in ranked[:top_k]]

# Enable by default in config
USE_RERANK_DEFAULT = True
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
```

#### 5.3 Answer Validation
**Timeline**: Week 3-4
**Effort**: Medium

```python
# clockify_rag/validation/
  ├── __init__.py
  ├── citation_checker.py        # Verify citations exist
  ├── hallucination_detector.py # Detect hallucination patterns
  └── confidence_calibrator.py   # Calibrate confidence scores

# clockify_rag/validation/citation_checker.py
class CitationChecker:
    """Validate citations in LLM answer."""

    def validate(self, answer: str, provided_chunks: list) -> dict:
        """Check if all citations exist in provided chunks."""
        # Extract citations: [id_123, id_456]
        citations = re.findall(r'\[(id_[^\]]+)\]', answer)

        # Check each citation
        chunk_ids = {c["id"] for c in provided_chunks}
        valid = [c for c in citations if c in chunk_ids]
        invalid = [c for c in citations if c not in chunk_ids]

        return {
            "valid": valid,
            "invalid": invalid,
            "pass": len(invalid) == 0
        }

# clockify_rag/validation/hallucination_detector.py
class HallucinationDetector:
    """Detect hallucination patterns in answer."""

    def detect(self, answer: str, provided_chunks: list) -> dict:
        """Check for common hallucination patterns."""
        hallucination_patterns = [
            r'in general',
            r'typically',
            r'usually',
            r'most (users|people)',
            r'it depends',
        ]

        # Check for speculation patterns
        speculation_matches = []
        for pattern in hallucination_patterns:
            if re.search(pattern, answer.lower()):
                speculation_matches.append(pattern)

        # Check for facts not in snippets (simple keyword matching)
        # More sophisticated approach: use entailment model

        return {
            "speculation_patterns": speculation_matches,
            "risk_level": "high" if len(speculation_matches) > 2 else "low"
        }

# Integration in answer_once()
def answer_once_validated(question: str, chunks, vecs_n, bm, ...):
    """Answer with validation."""
    # Retrieve and answer as usual
    answer, metadata = answer_once(question, chunks, vecs_n, bm, ...)

    # Validate answer
    citation_check = citation_checker.validate(answer, metadata["selected_chunks"])
    hallucination_check = hallucination_detector.detect(answer, metadata["selected_chunks"])

    # If validation fails, return refusal or flag answer
    if not citation_check["pass"] or hallucination_check["risk_level"] == "high":
        logger.warning(f"Answer validation failed: {citation_check}, {hallucination_check}")
        # Option 1: Return refusal
        return REFUSAL_STR, {"validated": False, "validation": {...}}
        # Option 2: Return flagged answer
        # return answer, {"validated": False, "validation": {...}}

    return answer, {"validated": True, "validation": {...}}
```

---

## Phase 6: Production Deployment & Scaling (Month 6)

### Objectives
- Deploy to production with monitoring
- Implement auto-scaling
- Add disaster recovery
- Performance optimization

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Load Balancer (nginx)                  │
│                    (SSL termination, rate limiting)          │
└─────────────────────────────────────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            │                                     │
┌───────────▼──────────┐             ┌──────────▼───────────┐
│   API Server 1       │             │   API Server 2       │
│   (FastAPI + uvicorn)│             │   (FastAPI + uvicorn)│
│   - 4 workers        │             │   - 4 workers        │
│   - Per-worker cache │             │   - Per-worker cache │
└───────────┬──────────┘             └──────────┬───────────┘
            │                                   │
            └─────────────┬──────────────────── ┘
                          │
            ┌─────────────▼──────────────┐
            │   Redis Cluster            │
            │   - Shared cache           │
            │   - Task queue (Celery)    │
            │   - Rate limit counters    │
            └─────────────┬──────────────┘
                          │
            ┌─────────────▼──────────────┐
            │   Worker Pool (Celery)     │
            │   - 10 workers             │
            │   - Async index building   │
            │   - Batch processing       │
            └─────────────┬──────────────┘
                          │
            ┌─────────────▼──────────────┐
            │   Storage (NFS/S3)         │
            │   - Per-tenant indexes     │
            │   - Knowledge bases        │
            │   - Backups                │
            └────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  Observability Stack                         │
│   - Prometheus (metrics)                                     │
│   - Grafana (dashboards)                                     │
│   - Jaeger (tracing)                                         │
│   - ELK Stack (logs)                                         │
└─────────────────────────────────────────────────────────────┘
```

### Auto-Scaling Configuration

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
spec:
  replicas: 2  # Initial replicas
  template:
    spec:
      containers:
      - name: api
        image: clockify-rag:6.0
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: redis://redis-service:6379
        - name: STORAGE_ROOT
          value: /mnt/storage
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## Technology Stack Evolution

### Current (v5.0)
- **Language**: Python 3.9+
- **HTTP**: requests
- **Numerical**: numpy
- **ML**: sentence-transformers, torch
- **Search**: BM25 (custom), FAISS
- **Testing**: pytest
- **Linting**: ruff, black, mypy

### Target (v6.0)
- **Language**: Python 3.11+ (performance improvements)
- **API**: FastAPI, uvicorn
- **HTTP**: httpx (async support)
- **Numerical**: numpy
- **ML**: sentence-transformers, torch, transformers
- **Search**: BM25 (custom), FAISS, Elasticsearch (optional)
- **Caching**: Redis
- **Task Queue**: Celery
- **Monitoring**: Prometheus, Grafana, Jaeger
- **Logging**: structlog, ELK Stack
- **Testing**: pytest, pytest-asyncio, pytest-xdist
- **Deployment**: Docker, Kubernetes

---

## Success Metrics

### Technical Metrics
- **Latency**: p95 < 100ms (down from ~200ms)
- **Throughput**: 100+ QPS per instance (up from ~10 QPS)
- **Availability**: 99.9% uptime
- **Cache Hit Rate**: >50%
- **Index Build Time**: <10 minutes for 10k docs (down from ~30 min)

### Quality Metrics
- **MRR**: ≥ 0.75 (up from unmeasured)
- **NDCG@10**: ≥ 0.70 (up from unmeasured)
- **Precision@5**: ≥ 0.65 (up from unmeasured)
- **Answer Accuracy**: ≥ 85% (measured via ground truth)

### Operational Metrics
- **Deployment Time**: <5 minutes (automated CI/CD)
- **Time to Detection (TTD)**: <5 minutes (monitoring alerts)
- **Time to Resolution (TTR)**: <15 minutes (runbooks + auto-remediation)

---

## Migration Path

### For Existing Deployments

#### Step 1: Backup Current State
```bash
# Backup indexes
tar -czf backup_$(date +%Y%m%d).tar.gz chunks.jsonl vecs_n.npy bm25.json faiss.index

# Backup configuration
cp clockify_support_cli_final.py clockify_support_cli_final.py.bak
```

#### Step 2: Install v6.0
```bash
# Update dependencies
pip install -r requirements-v6.txt

# Install new dependencies
pip install fastapi uvicorn redis celery prometheus-client opentelemetry-api
```

#### Step 3: Migrate to Modular Package
```bash
# Use migration script
python3 scripts/migrate_v5_to_v6.py

# Update imports
find . -name "*.py" -exec sed -i 's/from clockify_support_cli_final import/from clockify_rag import/g' {} +
```

#### Step 4: Deploy API
```bash
# Start Redis
docker run -d -p 6379:6379 redis:7

# Start Celery workers
celery -A clockify_rag.tasks.celery_app worker --loglevel=info

# Start API server
uvicorn clockify_rag.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Step 5: Gradual Rollout
```
Week 1: Deploy to staging, run full test suite
Week 2: Deploy to 10% of production traffic (canary)
Week 3: Deploy to 50% of production traffic
Week 4: Deploy to 100% of production traffic
```

---

## Risks & Mitigation

### Risk 1: Breaking Changes in API
**Mitigation**: Version API endpoints (`/v1/query`, `/v2/query`), maintain v1 for 6 months

### Risk 2: Performance Degradation
**Mitigation**: Comprehensive benchmarking before each deployment, automated rollback

### Risk 3: Data Migration Failures
**Mitigation**: Atomic migrations, backup before migration, rollback scripts

### Risk 4: Multi-Tenancy Bugs
**Mitigation**: Extensive integration tests, tenant isolation validation, audit logging

### Risk 5: Scalability Bottlenecks
**Mitigation**: Load testing at 2x expected load, auto-scaling policies, capacity planning

---

## Conclusion

This architecture vision provides a clear roadmap from the current v5.0 (modular with plugin architecture) to a production-grade, scalable v6.0 platform supporting:

✅ **Multi-tenancy** with 100+ tenants
✅ **High concurrency** with 100+ QPS per instance
✅ **Continuous evaluation** with automated metrics
✅ **REST API** for programmatic access
✅ **Distributed indexing** for large knowledge bases
✅ **Full observability** with metrics, tracing, and logs

**Estimated Timeline**: 6 months (with 2-3 full-time engineers)
**Estimated Effort**: ~1,500 engineering hours
**Expected ROI**: 10x improvement in scalability, 5x improvement in quality, 3x reduction in operational overhead

**Next Steps**:
1. Review and approve architecture vision
2. Prioritize phases based on business needs
3. Assign engineering resources
4. Begin Phase 1 (Modularization & Thread Safety)

---

**Document Status**: DRAFT for Review
**Author**: Claude (Senior ML/RAG Engineer)
**Date**: 2025-11-06
**Version**: 1.0
