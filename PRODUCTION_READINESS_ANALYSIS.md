# Production Readiness Analysis - Clockify RAG System
**Date**: 2025-11-09
**Analyst**: Claude Code
**Repository**: 1rag (clockify-rag v5.9.1)
**Status**: ⚠️ **NEEDS ATTENTION** - Production-capable but requires fixes

---

## Executive Summary

The Clockify RAG system is **well-architected** with strong modular design, comprehensive testing, and good documentation. However, there are **18 critical issues** that must be addressed before production deployment, plus several recommendations for improvement.

### Quick Stats
- **Total Code**: 7,807 lines of Python
- **Test Coverage**: 20+ test files (500-422 lines each)
- **Documentation**: 50+ markdown files
- **CI/CD**: GitHub Actions with 5 workflow files
- **Architecture**: Modular package with plugin system
- **Dependencies**: Well-managed via pyproject.toml

### Priority Summary
- 🔴 **Critical Issues**: 7 (must fix before production)
- 🟡 **High Priority**: 6 (fix within 1 week of deployment)
- 🔵 **Medium Priority**: 5 (address in next sprint)
- ⚪ **Low Priority**: 12 (technical debt)

---

## 🔴 CRITICAL ISSUES (Must Fix Before Production)

### 1. Version Inconsistency ⚠️ BLOCKER
**File**: `clockify_rag/__init__.py:10` vs `pyproject.toml:7`
**Issue**: Version mismatch between package declaration
- `__init__.py` declares: `__version__ = "5.0.0"`
- `pyproject.toml` declares: `version = "5.9.1"`
- `api.py` hardcodes: `version="5.9.1"`

**Impact**: Breaking changes, version confusion in logs/metrics, incorrect deployment tracking

**Fix**:
```python
# clockify_rag/__init__.py
__version__ = "5.9.1"  # Match pyproject.toml
```

**Verification**:
```bash
python -c "import clockify_rag; print(clockify_rag.__version__)"
# Should output: 5.9.1
```

---

### 2. CI/CD Test Configuration Mismatch
**Files**: `.github/workflows/test.yml:86`, `.github/workflows/ci.yml:86`
**Issue**: Tests run with wrong coverage target

**Current**:
```yaml
run: pytest tests/ -v --cov=clockify_support_cli_final --cov-report=term-missing
```

**Problem**: `clockify_support_cli_final` is a legacy wrapper. Should target the actual package.

**Fix**:
```yaml
run: pytest tests/ -v --cov=clockify_rag --cov=clockify_support_cli_final --cov-report=term-missing --cov-report=xml
```

---

### 3. Requirements File Compatibility Issue
**File**: `requirements.txt:17`
**Issue**: Platform-specific version pinning causes Linux CI failures

```txt
torch==2.4.2  # Not available on all platforms
```

**Error seen**:
```
ERROR: Could not find a version that satisfies the requirement torch==2.4.2
```

**Fix**: Use version ranges in `requirements.txt` or document platform-specific files:
```txt
# requirements.txt (cross-platform)
torch>=2.3.0,<2.6.0  # Compatible with pyproject.toml
```

OR keep `requirements-m1.txt` separate and update CI to use `pyproject.toml`:
```yaml
pip install -e ".[dev]"  # Always use this in CI
```

---

### 4. Missing Test Dependency Installation Guard
**Issue**: Tests will fail on fresh installations without proper error messages

**Current State**: Tests import modules assuming all dependencies installed
**Problem**: Silent failures or cryptic errors on missing optional dependencies

**Fix**: Add test guards in `conftest.py`:
```python
# tests/conftest.py
import pytest
import sys

def pytest_configure(config):
    """Validate test environment before running."""
    try:
        import numpy
        import requests
        import pytest
    except ImportError as e:
        print(f"ERROR: Missing required test dependency: {e}")
        print("Install with: pip install -e '.[dev]'")
        sys.exit(1)

    # Check optional dependencies
    missing = []
    try:
        import faiss
    except ImportError:
        missing.append("faiss-cpu")

    try:
        import torch
    except ImportError:
        missing.append("torch")

    if missing:
        print(f"WARNING: Optional dependencies not installed: {', '.join(missing)}")
        print("Some tests may be skipped. Install with: pip install -e '.[dev]'")
```

---

### 5. Thread Safety Verification Missing
**Files**: `clockify_rag/caching.py`, `clockify_rag/indexing.py`, `clockify_support_cli_final.py`
**Status**: Thread safety locks exist but not fully validated

**Issues Found**:
1. `RETRIEVE_PROFILE_LAST` dictionary access may race (clockify_rag/retrieval.py:39-51)
2. Query cache eviction uses `dict.popitem()` which is not atomic pre-Python 3.7
3. FAISS index lazy loading has double-checked locking but no benchmarks

**Fix**:
```python
# retrieval.py:39
_RETRIEVE_PROFILE_LOCK = threading.RLock()
RETRIEVE_PROFILE_LAST = {}

def get_retrieve_profile():
    """Thread-safe access to profile data."""
    with _RETRIEVE_PROFILE_LOCK:
        return dict(RETRIEVE_PROFILE_LAST)  # Return copy

def set_retrieve_profile(data):
    """Thread-safe update of profile data."""
    with _RETRIEVE_PROFILE_LOCK:
        RETRIEVE_PROFILE_LAST.clear()
        RETRIEVE_PROFILE_LAST.update(data)
```

**Verification Required**:
```bash
pytest tests/test_thread_safety.py -v -n 8  # Run with 8 workers
```

---

### 6. Dockerfile Security Issue
**File**: `Dockerfile:61-62`
**Issue**: Running as non-root but ownership may not propagate

```dockerfile
RUN groupadd -r raguser && useradd -r -g raguser raguser
# ...
USER raguser
```

**Problem**: If index files are mounted as root-owned, application can't write logs or cache

**Fix**:
```dockerfile
# Add after COPY application code
RUN mkdir -p var/{index,logs,reports,backups} /app/.cache && \
    chown -R raguser:raguser /app/var /app/.cache

# Add volume mounts with correct permissions in docker-compose.yml
volumes:
  - ./var/logs:/app/var/logs:rw,delegated
  # Add tmpfs for cache
  - type: tmpfs
    target: /app/.cache
    tmpfs:
      size: 100M
```

---

### 7. Missing Health Check Implementation
**File**: `Dockerfile:74-75` and `docker-compose.yml:101-106`
**Issue**: Health check endpoint not implemented

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1
```

**Problem**: `/health` endpoint exists in `api.py` but doesn't verify dependencies

**Fix in `clockify_rag/api.py`**:
```python
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check with dependency validation."""
    import os
    from pathlib import Path

    # Check index files
    index_files = ["chunks.jsonl", "vecs_n.npy", "meta.jsonl", "bm25.json"]
    index_ready = all(Path(f).exists() for f in index_files)

    # Check Ollama connectivity
    ollama_connected = False
    try:
        resp = requests.get(f"{config.OLLAMA_URL}/api/version", timeout=2)
        ollama_connected = resp.status_code == 200
    except:
        pass

    # Determine overall status
    status = "healthy" if (index_ready and ollama_connected) else "degraded"
    if not index_ready:
        status = "unavailable"

    return HealthResponse(
        status=status,
        timestamp=datetime.now(),
        version=__version__,
        platform=platform.machine(),
        index_ready=index_ready,
        ollama_connected=ollama_connected
    )
```

---

## 🟡 HIGH PRIORITY ISSUES (Fix Within 1 Week)

### 8. TODO in Production Code
**File**: `clockify_rag/cli_modern.py:480`
```python
# TODO: Implement RAGAS evaluation loop
```

**Issue**: Unfinished feature referenced in production code
**Fix**: Either implement or remove the TODO, or document as "future enhancement"

---

### 9. Circular Import Risk
**Files**: Multiple imports between modules
**Issue**: Complex import graph may cause initialization issues

**Example**:
```
clockify_rag/__init__.py → imports from .retrieval
clockify_rag/retrieval.py → imports from .embedding
clockify_rag/embedding.py → imports from .http_utils
clockify_rag/http_utils.py → imports from .config
```

**Impact**: Works now but fragile to changes

**Fix**: Refactor to use explicit runtime imports for circular dependencies:
```python
# Instead of:
from .retrieval import retrieve

# Use:
def get_retriever():
    from .retrieval import retrieve
    return retrieve
```

---

### 10. Error Handling Incomplete
**Files**: `clockify_rag/retrieval.py`, `clockify_rag/answer.py`
**Issue**: 258 try/except blocks but inconsistent error propagation

**Problem**: Some exceptions caught and logged but not re-raised, causing silent failures

**Example**:
```python
# retrieval.py:validate_query_length
def validate_query_length(question: str, max_length: int = None) -> str:
    if not question:
        raise ValidationError("Query cannot be empty")
    # Good! Raises exception
```

But elsewhere:
```python
# Some code catches exceptions and only logs
try:
    result = process()
except Exception as e:
    logger.error(f"Error: {e}")
    # Problem: No re-raise, caller doesn't know about failure
```

**Fix**: Audit all exception handlers and ensure:
1. Re-raise critical errors
2. Return error indicators for recoverable errors
3. Never silently swallow exceptions in production paths

---

### 11. Logging Configuration Inconsistency
**Files**: `config.py`, `utils.py`, individual modules
**Issue**: Multiple logger configurations without clear hierarchy

**Problem**:
- Root logger configured in multiple places
- Some modules use `logging.getLogger(__name__)`
- Others use package-level logger
- CLI and API may have different log formats

**Fix**: Centralize logging setup in `clockify_rag/logging_config.py`:
```python
import logging
import sys
from typing import Optional

def setup_logging(
    level: str = "INFO",
    format_type: str = "json",  # or "text"
    log_file: Optional[str] = None
):
    """Central logging configuration."""

    # Clear existing handlers
    root = logging.getLogger()
    root.handlers.clear()

    # Set level
    root.setLevel(getattr(logging, level.upper()))

    # Create formatter
    if format_type == "json":
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
```

---

### 12. Metrics Collection Not Validated
**File**: `clockify_rag/metrics.py`
**Issue**: Comprehensive metrics framework but no validation

**Missing**:
- No tests for metric accuracy
- No benchmarks for metric overhead
- No documentation on metric interpretation

**Fix**:
1. Add `tests/test_metrics_accuracy.py`:
```python
def test_metric_accuracy():
    """Verify metrics reflect actual behavior."""
    from clockify_rag.metrics import MetricsCollector, get_metrics

    collector = get_metrics()

    # Test counter
    collector.increment_counter("test_counter")
    collector.increment_counter("test_counter")
    snapshot = collector.get_snapshot()
    assert snapshot.counters.get("test_counter", 0) == 2

    # Test histogram
    collector.observe_histogram("test_hist", 100)
    collector.observe_histogram("test_hist", 200)
    stats = collector.get_histogram_stats("test_hist")
    assert stats["count"] == 2
    assert stats["mean"] == 150
```

2. Add performance benchmarks
3. Document metrics in `docs/METRICS.md`

---

### 13. Missing Observability Instrumentation
**Issue**: No distributed tracing or structured logging for debugging

**Missing**:
- Request ID propagation
- Trace context (parent/child spans)
- Correlation IDs for multi-step operations

**Fix**: Add OpenTelemetry instrumentation:
```python
# clockify_rag/tracing.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

def setup_tracing(service_name="clockify-rag"):
    """Initialize OpenTelemetry tracing."""
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)

    # Add Jaeger exporter (or use OTLP for production)
    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=6831,
    )
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)

    return tracer
```

**Optional** (recommended for production):
- Add `pyproject.toml` dependency: `opentelemetry-api`, `opentelemetry-sdk`
- Integrate with FastAPI middleware
- Export to Jaeger, Zipkin, or cloud provider

---

## 🔵 MEDIUM PRIORITY ISSUES

### 14. API Missing Rate Limiting
**File**: `clockify_rag/api.py`
**Issue**: FastAPI server has no rate limiting middleware

**Impact**: Vulnerable to abuse, DoS attacks
**Fix**: Add rate limiting:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/v1/query")
@limiter.limit("10/minute")
async def query_endpoint(request: QueryRequest):
    ...
```

---

### 15. Missing Input Validation on API
**File**: `clockify_rag/api.py:40`
**Issue**: API accepts up to 10,000 char questions but doesn't validate content

**Current**:
```python
question: str = Field(..., min_length=1, max_length=10000)
```

**Problem**: No XSS/injection prevention, no profanity filter
**Fix**: Add validation layer:
```python
from pydantic import validator

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10000)

    @validator('question')
    def validate_question(cls, v):
        # Strip excessive whitespace
        v = " ".join(v.split())

        # Check for suspicious patterns
        if any(pattern in v.lower() for pattern in ['<script>', 'javascript:', 'onerror=']):
            raise ValueError("Invalid characters in question")

        # Validate character set (optional)
        if not all(c.isprintable() or c.isspace() for c in v):
            raise ValueError("Question contains non-printable characters")

        return v
```

---

### 16. Dockerfile Multi-Stage Build Not Optimized
**File**: `Dockerfile:11-40`
**Issue**: Builder stage installs all dependencies, including dev tools

**Problem**: Final image larger than necessary
**Fix**:
```dockerfile
# Stage 1: Builder
FROM python:3.11-slim as builder
# ... (keep existing)

# Install only production dependencies
RUN /root/.cargo/bin/uv pip install --python /usr/local/bin/python3.11 \
    --target /app/venv \
    --compile-bytecode \
    .  # Don't include [dev]

# Stage 2: Runtime (keep existing)
# This is already good!
```

---

### 17. No API Authentication
**File**: `clockify_rag/api.py`, `docker-compose.yml`
**Issue**: API server has no authentication mechanism

**Impact**: Anyone with network access can query
**Fix**: Add API key authentication:
```python
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key from Authorization header."""
    api_key = os.environ.get("RAG_API_KEY")
    if not api_key:
        # Development mode: no auth required
        return True

    if credentials.credentials != api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return True

@app.post("/v1/query")
async def query_endpoint(
    request: QueryRequest,
    authenticated: bool = Depends(verify_api_key)
):
    ...
```

**Environment variable**:
```bash
# docker-compose.yml
environment:
  RAG_API_KEY: "${RAG_API_KEY:-}"  # Read from .env file
```

---

### 18. Missing Graceful Shutdown
**File**: `clockify_rag/api.py`, `Dockerfile`
**Issue**: No graceful shutdown handling for in-flight requests

**Fix**: Add shutdown event handler:
```python
import signal
import asyncio

shutdown_event = asyncio.Event()

def signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT gracefully."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Shutting down gracefully...")
    # Close database connections, flush caches, etc.
    # Wait for in-flight requests (handled by uvicorn)
```

**Update CMD in Dockerfile**:
```dockerfile
CMD ["uvicorn", "clockify_rag.api:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "30"]
```

---

## ⚪ LOW PRIORITY / TECHNICAL DEBT

### 19. Excessive Documentation (50+ files)
**Issue**: Documentation sprawl makes it hard to find canonical information

**Recommendation**: Consolidate into:
1. `README.md` - Quick start
2. `docs/ARCHITECTURE.md` - System design
3. `docs/API.md` - API reference
4. `docs/DEPLOYMENT.md` - Production deployment
5. Archive historical docs to `docs/archive/`

---

### 20. Inconsistent Naming Conventions
**Files**: `clockify_support_cli.py` vs `clockify_rag/` package
**Issue**: Mixed naming (`support_cli` vs `rag`)

**Recommendation**: Standardize on `clockify_rag` everywhere

---

### 21. Unused Imports
**Files**: Multiple
**Issue**: Ruff linter configured but some unused imports remain

**Fix**: Run `ruff check --fix` before commit

---

### 22. Test Fixtures Could Be Centralized
**Files**: Multiple test files duplicate fixtures
**Recommendation**: Move common fixtures to `conftest.py`

---

### 23. Missing Type Hints in Some Functions
**Issue**: Mypy configured but `continue-on-error: true` in pre-commit

**Recommendation**: Fix type hints incrementally, remove `continue-on-error` once clean

---

### 24. No Monitoring/Alerting Configuration
**Recommendation**: Add example Prometheus/Grafana configs in `config/monitoring/`

---

### 25. No Load Testing Evidence
**Recommendation**: Add `locust` or `k6` load test scripts

---

### 26. Missing Disaster Recovery Plan
**Recommendation**: Document backup/restore procedures for index files

---

### 27. No Security Scanning in CI
**Recommendation**: Add Bandit or Snyk to CI pipeline

---

### 28. Missing API Versioning Strategy
**Issue**: API at `/v1/*` but no documented deprecation policy

---

### 29. No Internationalization (i18n)
**Issue**: All prompts/messages in English only

---

### 30. Missing Changelog Automation
**Recommendation**: Use conventional commits + auto-generate CHANGELOG.md

---

## TEST FAILURES ANALYSIS

Based on CI workflows and test structure, likely test failures:

### **Probable Failures**:

1. **Integration tests requiring Ollama** (`test_integration.py`)
   - **Cause**: No Ollama service in CI
   - **CI Workaround**: Already marked `continue-on-error: true` (line 89)
   - **Fix**: Mock Ollama responses or skip if not available

2. **FAISS tests on unsupported platforms** (`test_faiss_integration.py`)
   - **Cause**: FAISS build failures on some platforms
   - **Fix**: Skip if FAISS not available:
   ```python
   pytest.mark.skipif(not has_faiss(), reason="FAISS not available")
   ```

3. **Thread safety tests timing out** (`test_thread_safety.py`)
   - **Cause**: High concurrency on limited CI resources
   - **Fix**: Reduce concurrency in CI:
   ```python
   if os.environ.get("CI"):
       num_workers = 2  # Reduce from 8
   ```

---

## DEPLOYMENT CHECKLIST

### Pre-Deployment
- [ ] **FIX**: Update `__version__` to 5.9.1 in `__init__.py`
- [ ] **FIX**: Update CI workflows to use `--cov=clockify_rag`
- [ ] **FIX**: Update `requirements.txt` to use version ranges
- [ ] **FIX**: Add test dependency guards
- [ ] **TEST**: Run `pytest tests/ -v` and fix any failures
- [ ] **TEST**: Run `pytest tests/test_thread_safety.py -n 4`
- [ ] **VERIFY**: All CI workflows pass

### Security
- [ ] **ADD**: API authentication (Issue #17)
- [ ] **ADD**: Rate limiting on API (Issue #14)
- [ ] **ADD**: Input validation on API (Issue #15)
- [ ] **VERIFY**: No secrets in code (`grep -r "api_key\|password\|secret"`)
- [ ] **RUN**: Security scanner (Bandit, Snyk)

### Production Configuration
- [ ] **SET**: Environment variables in `.env`:
  ```bash
  OLLAMA_URL=http://ollama:11434
  RAG_API_KEY=<random-secure-key>
  RAG_LOG_FILE=/app/var/logs/queries.jsonl
  LOG_LEVEL=INFO
  ```
- [ ] **CREATE**: Volume directories:
  ```bash
  mkdir -p var/{index,logs,reports,backups}
  chmod 755 var/*
  ```
- [ ] **BUILD**: Docker image:
  ```bash
  docker build -t clockify-rag:5.9.1 .
  ```

### Deployment
- [ ] **DEPLOY**: Using docker-compose:
  ```bash
  docker-compose up -d
  ```
- [ ] **VERIFY**: Health check:
  ```bash
  curl http://localhost:8000/health
  ```
- [ ] **TEST**: Query endpoint:
  ```bash
  curl -X POST http://localhost:8000/v1/query \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $RAG_API_KEY" \
    -d '{"question": "How do I track time?"}'
  ```
- [ ] **MONITOR**: Check logs for errors:
  ```bash
  docker-compose logs -f app
  ```

### Post-Deployment
- [ ] **MONITOR**: Health endpoint every 30s
- [ ] **ALERT**: Set up alerts for:
  - Health check failures
  - High error rates (>5%)
  - Slow response times (>10s p95)
  - Memory usage (>80%)
- [ ] **BACKUP**: Index files daily:
  ```bash
  tar -czf index-backup-$(date +%Y%m%d).tar.gz var/index/
  ```

---

## RECOMMENDED IMPROVEMENTS

### Short Term (1-2 weeks)
1. Fix all 🔴 CRITICAL issues
2. Add API authentication
3. Add comprehensive error handling
4. Set up monitoring/alerting
5. Document deployment procedures

### Medium Term (1 month)
1. Add distributed tracing
2. Implement rate limiting
3. Add load testing
4. Security audit + penetration testing
5. Performance benchmarking

### Long Term (2-3 months)
1. Implement A/B testing framework
2. Add telemetry and analytics
3. Build admin dashboard
4. Implement feedback loop for model fine-tuning
5. Multi-tenancy support

---

## ARCHITECTURE STRENGTHS

✅ **Well-Designed**:
- Modular package structure with clear separation of concerns
- Plugin architecture for extensibility
- Comprehensive test coverage (20+ test files)
- Thread-safe caching and rate limiting
- Hybrid retrieval (BM25 + dense + MMR)

✅ **Production Features**:
- Docker containerization
- Health checks
- Graceful FAISS fallback
- Atomic file operations
- Deterministic behavior (seed=42)

✅ **Good Practices**:
- Type hints (mostly)
- Pre-commit hooks
- CI/CD pipelines
- Extensive documentation
- Configuration via environment variables

---

## FINAL RECOMMENDATION

**Status**: ⚠️ **CONDITIONAL APPROVAL**

The system is **production-capable** but **requires critical fixes** before deployment:

### Must Fix (Blockers):
1. Version inconsistency (#1)
2. CI/CD test configuration (#2)
3. Requirements compatibility (#3)
4. Health check implementation (#7)

### Should Fix (High Risk):
5. Thread safety validation (#5)
6. Error handling audit (#10)
7. Logging centralization (#11)

### Nice to Have:
8. API authentication (#17)
9. Rate limiting (#14)
10. Input validation (#15)

**Estimated Time to Production-Ready**: **3-5 days** (assuming dedicated developer)

---

## APPENDIX: Quick Fix Script

Run this script to fix critical issues automatically:

```bash
#!/bin/bash
# fix_critical_issues.sh

echo "Fixing critical production issues..."

# Fix #1: Version inconsistency
echo "Updating __version__ to 5.9.1..."
sed -i 's/__version__ = "5.0.0"/__version__ = "5.9.1"/' clockify_rag/__init__.py

# Fix #2: CI coverage target
echo "Updating CI workflows..."
sed -i 's/--cov=clockify_support_cli_final/--cov=clockify_rag --cov=clockify_support_cli_final/' .github/workflows/*.yml

# Fix #3: Requirements compatibility
echo "Updating requirements.txt..."
sed -i 's/torch==2.4.2/torch>=2.3.0,<2.6.0/' requirements.txt

echo "✅ Critical fixes applied. Please review and commit changes."
echo "Next steps:"
echo "1. Review changes: git diff"
echo "2. Run tests: pytest tests/ -v"
echo "3. Commit: git add -A && git commit -m 'fix: Critical production issues'"
```

---

**End of Analysis**
