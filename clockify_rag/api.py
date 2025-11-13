"""FastAPI server for Clockify RAG system.

Provides REST API endpoints:
- GET /health: Health check
- GET /v1/config: Current configuration
- POST /v1/query: Submit a question
- POST /v1/ingest: Trigger index build
- GET /v1/metrics: System metrics
"""

import asyncio
import json
import logging
import os
import platform
import signal
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any, List

import typer
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, validator

from . import config
from .answer import answer_once
from .caching import get_rate_limiter
from .cli import ensure_index_ready
from .indexing import build
from .utils import check_ollama_connectivity
from .exceptions import ValidationError

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency when JWT auth disabled
    import jwt
except ModuleNotFoundError:  # pragma: no cover - dependency is optional
    jwt = None


async def validate_request_credentials(request: Request) -> Dict[str, Any]:
    """Validate request credentials based on configured auth strategy."""

    mode = (config.API_AUTH_MODE or "none").lower()
    if mode in {"", "none"}:
        return {"method": "none"}

    if mode == "api_key":
        if not config.API_ALLOWED_KEYS:
            logger.error("API key authentication enabled but no keys loaded")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication misconfigured",
            )

        header_name = config.API_KEY_HEADER or "x-api-key"
        provided_key = request.headers.get(header_name)
        if not provided_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key",
            )

        if provided_key not in config.API_ALLOWED_KEYS:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key",
            )

        return {"method": "api_key", "principal": provided_key}

    if mode == "jwt":
        if not config.API_JWT_SECRET:
            logger.error("JWT authentication enabled but secret not configured")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication misconfigured",
            )

        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Authorization header",
            )

        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Authorization header",
            )

        if jwt is None:
            logger.error("JWT authentication requested but PyJWT is not installed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="JWT authentication unavailable",
            )

        try:
            decoded = jwt.decode(token, config.API_JWT_SECRET, algorithms=config.API_JWT_ALGORITHMS or None)
        except Exception as exc:  # pragma: no cover - PyJWT supplies detailed errors
            expired_error = getattr(jwt, "ExpiredSignatureError", tuple())
            invalid_error = getattr(jwt, "InvalidTokenError", tuple())
            if expired_error and isinstance(exc, expired_error):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token expired",
                ) from exc
            if invalid_error and isinstance(exc, invalid_error):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid token",
                ) from exc
            logger.error("Unexpected JWT validation error: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication misconfigured",
            ) from exc

        return {"method": "jwt", "principal": decoded.get("sub"), "claims": decoded}

    logger.error("Unsupported authentication mode: %s", config.API_AUTH_MODE)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Authentication misconfigured",
    )

# ============================================================================
# Pydantic Models
# ============================================================================


class QueryRequest(BaseModel):
    """Request body for /v1/query endpoint."""

    question: str = Field(..., min_length=1, max_length=10000, description="Question to answer")
    top_k: Optional[int] = Field(15, ge=1, le=100, description="Number of chunks to retrieve")
    pack_top: Optional[int] = Field(8, ge=1, le=50, description="Number of chunks in context")
    threshold: Optional[float] = Field(0.25, ge=0.0, le=1.0, description="Minimum similarity")
    debug: Optional[bool] = Field(False, description="Include debug information")

    @validator('question')
    def validate_question(cls, v):
        """Validate and sanitize question input.

        Prevents XSS, injection attacks, and other malicious input.
        """
        # Strip excessive whitespace
        v = " ".join(v.split())

        if not v:
            raise ValueError("Question cannot be empty after whitespace removal")

        # Check for suspicious patterns (basic XSS prevention)
        suspicious_patterns = [
            '<script',
            'javascript:',
            'onerror=',
            'onload=',
            '<iframe',
            'eval(',
            'expression(',
        ]

        v_lower = v.lower()
        for pattern in suspicious_patterns:
            if pattern in v_lower:
                raise ValueError(f"Invalid content detected in question")

        # Ensure only printable characters (allow unicode for i18n)
        if not all(c.isprintable() or c.isspace() for c in v):
            raise ValueError("Question contains non-printable characters")

        return v


class QueryResponse(BaseModel):
    """Response body for /v1/query endpoint."""

    question: str
    answer: str
    confidence: Optional[float] = None
    sources: List[int] = Field(default_factory=list, description="Chunk IDs used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata from answer pipeline")
    routing: Optional[Dict[str, Any]] = Field(default=None, description="Routing recommendation metadata")
    timestamp: datetime
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Response body for /v1/health endpoint."""

    status: str
    timestamp: datetime
    version: str
    platform: str
    index_ready: bool
    ollama_connected: bool


class ConfigResponse(BaseModel):
    """Response body for /v1/config endpoint."""

    ollama_url: str
    gen_model: str
    emb_model: str
    chunk_size: int
    top_k: int
    pack_top: int
    threshold: float


class IngestRequest(BaseModel):
    """Request body for /v1/ingest endpoint."""

    input_file: Optional[str] = Field(None, description="Input markdown file")
    force: Optional[bool] = Field(False, description="Force rebuild")


class IngestResponse(BaseModel):
    """Response body for /v1/ingest endpoint."""

    status: str
    message: str
    timestamp: datetime
    index_ready: bool


# ============================================================================
# FastAPI Application
# ============================================================================


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Clockify RAG API",
        description="Production-ready RAG system with hybrid retrieval",
        version="5.9.1",
    )

    # Add CORS middleware if enabled
    if config.ALPHA_HYBRID is not None:  # Placeholder check
        origins = config.RAG_API_ALLOWED_ORIGINS
        allow_credentials = "*" not in origins
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Global state for index
    app.state.chunks = None
    app.state.vecs_n = None
    app.state.bm = None
    app.state.hnsw = None
    app.state.index_ready = False
    app.state.index_lock = threading.RLock()

    @app.on_event("startup")
    async def startup_event():
        """Load index on startup."""
        try:
            logger.info("Loading index on startup...")
            result = ensure_index_ready(retries=2)
            if result:
                chunks, vecs_n, bm, hnsw = result
                with app.state.index_lock:
                    app.state.chunks = chunks
                    app.state.vecs_n = vecs_n
                    app.state.bm = bm
                    app.state.hnsw = hnsw
                    app.state.index_ready = True
                logger.info(f"Index loaded: {len(chunks)} chunks")
            else:
                logger.warning("Index not ready at startup")
        except Exception as e:
            logger.error(f"Failed to load index at startup: {e}")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Graceful shutdown handler.

        Performs cleanup tasks:
        - Flushes any pending logs
        - Closes database connections (if any)
        - Saves metrics/cache (if configured)
        - Allows in-flight requests to complete (handled by uvicorn)
        """
        logger.info("Initiating graceful shutdown...")

        # Clear index from memory (helps with clean shutdown)
        with app.state.index_lock:
            app.state.chunks = None
            app.state.vecs_n = None
            app.state.bm = None
            app.state.hnsw = None
            app.state.index_ready = False

        logger.info("Graceful shutdown complete")

    # ========================================================================
    # Health Check Endpoint
    # ========================================================================

    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Enhanced health check endpoint with dependency validation.

        Returns system status, index readiness, and Ollama connectivity.
        Status levels:
        - healthy: Index ready and Ollama connected
        - degraded: Index ready but Ollama unavailable (can serve cached queries)
        - unavailable: Index not ready (cannot serve any queries)
        """
        from . import __version__
        from pathlib import Path

        # Check index files exist (belt-and-suspenders with app.state)
        index_files_exist = all(
            Path(f).exists() for f in ["chunks.jsonl", "vecs_n.npy", "meta.jsonl", "bm25.json"]
        )
        with app.state.index_lock:
            index_ready = app.state.index_ready
        index_ready = index_ready and index_files_exist

        # Check Ollama connectivity with short timeout
        ollama_ok = False
        try:
            check_ollama_connectivity(config.OLLAMA_URL, timeout=2)
            ollama_ok = True
        except Exception as e:
            # Ollama connectivity failure is acceptable for health check
            # (allows graceful degradation), but log for debugging
            logger.debug(f"Ollama health check failed: {e}")

        # Determine overall status
        if not index_ready:
            status = "unavailable"
        elif index_ready and ollama_ok:
            status = "healthy"
        else:
            status = "degraded"

        return HealthResponse(
            status=status,
            timestamp=datetime.now(),
            version=__version__,
            platform=f"{platform.system()} {platform.machine()}",
            index_ready=index_ready,
            ollama_connected=ollama_ok,
        )

    @app.get("/v1/health", response_model=HealthResponse)
    async def health_check_v1() -> HealthResponse:
        """Health check endpoint (v1 API)."""
        return await health_check()

    # ========================================================================
    # Configuration Endpoint
    # ========================================================================

    @app.get("/v1/config", response_model=ConfigResponse)
    async def get_config() -> ConfigResponse:
        """Get current configuration."""
        return ConfigResponse(
            ollama_url=config.OLLAMA_URL,
            gen_model=config.GEN_MODEL,
            emb_model=config.EMB_MODEL,
            chunk_size=config.CHUNK_CHARS,
            top_k=config.DEFAULT_TOP_K,
            pack_top=config.DEFAULT_PACK_TOP,
            threshold=config.DEFAULT_THRESHOLD,
        )

    # ========================================================================
    # Query Endpoint
    # ========================================================================

    @app.post("/v1/query", response_model=QueryResponse)
    async def submit_query(
        request: QueryRequest,
        raw_request: Request,
        credentials: Dict[str, Any] = Depends(validate_request_credentials),
    ) -> QueryResponse:
        """Submit a question and get an answer.

        This endpoint uses the RAG system to retrieve relevant context
        and generate an answer using the LLM.

        Args:
            request: QueryRequest with question and parameters

        Returns:
            QueryResponse with answer, confidence, and sources

        Raises:
            HTTPException: If index not ready or query fails
        """
        with app.state.index_lock:
            index_ready = app.state.index_ready
            chunks = app.state.chunks
            vecs_n = app.state.vecs_n
            bm = app.state.bm
            hnsw = app.state.hnsw

        if not index_ready:
            raise HTTPException(
                status_code=503, detail="Index not ready. Run /v1/ingest first or wait for startup."
            )

        limiter = get_rate_limiter()
        principal = credentials.get("principal") if isinstance(credentials, dict) else None
        client_host = raw_request.client.host if raw_request.client else None
        limiter_identity = principal or client_host or "api-anonymous"

        if not limiter.allow_request(limiter_identity):
            wait_seconds = limiter.wait_time(limiter_identity)
            detail = f"Rate limit exceeded. Retry after {wait_seconds:.0f} seconds."
            raise HTTPException(status_code=429, detail=detail)

        try:
            start_time = time.time()

            result = await run_in_threadpool(
                answer_once,
                request.question,
                chunks,
                vecs_n,
                bm,
                top_k=request.top_k,
                pack_top=request.pack_top,
                threshold=request.threshold,
                hnsw=hnsw,
            )

            elapsed_ms = (time.time() - start_time) * 1000

            return QueryResponse(
                question=request.question,
                answer=result.get("answer", ""),
                confidence=result.get("confidence"),
                sources=result.get("selected_chunks", [])[:5],  # Top 5 sources
                metadata=result.get("metadata", {}) or {},
                routing=result.get("routing"),
                timestamp=datetime.now(),
                processing_time_ms=elapsed_ms,
            )

        except ValidationError as exc:
            logger.info("Validation error: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as e:
            logger.error(f"Query error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

    # ========================================================================
    # Ingest Endpoint
    # ========================================================================

    secure_ingest_modes = {"api_key", "jwt"}
    auth_mode = (config.API_AUTH_MODE or "none").strip().lower()

    if auth_mode in secure_ingest_modes:

        @app.post("/v1/ingest", response_model=IngestResponse)
        async def trigger_ingest(
            request: IngestRequest,
            background_tasks: BackgroundTasks,
            _: Dict[str, Any] = Depends(validate_request_credentials),
        ) -> IngestResponse:
            """Trigger index build/rebuild.

            Starts a background task to build the index from the knowledge base.

            Args:
                request: IngestRequest with input file and options
                background_tasks: FastAPI background tasks

            Returns:
                IngestResponse with status

            Note:
                Build happens asynchronously. Check /health to verify completion.
            """
            input_file = request.input_file or "knowledge_full.md"

            if not os.path.exists(input_file):
                raise HTTPException(status_code=404, detail=f"Input file not found: {input_file}")

            def do_ingest():
                """Background task to build index."""
                try:
                    logger.info(f"Starting ingest from {input_file}")
                    build(input_file, retries=2)

                    result = ensure_index_ready(retries=2)
                    if not result:
                        raise RuntimeError("Index artifacts missing after build")

                    chunks, vecs_n, bm, hnsw = result
                    with app.state.index_lock:
                        app.state.chunks = chunks
                        app.state.vecs_n = vecs_n
                        app.state.bm = bm
                        app.state.hnsw = hnsw
                        app.state.index_ready = True
                    logger.info("Ingest completed successfully")
                except Exception as e:
                    logger.error(f"Ingest failed: {e}", exc_info=True)
                    with app.state.index_lock:
                        app.state.chunks = None
                        app.state.vecs_n = None
                        app.state.bm = None
                        app.state.hnsw = None
                        app.state.index_ready = False

            background_tasks.add_task(do_ingest)

            with app.state.index_lock:
                index_ready = app.state.index_ready

            return IngestResponse(
                status="processing",
                message=f"Index build started in background from {input_file}",
                timestamp=datetime.now(),
                index_ready=index_ready,
            )

    else:
        logger.warning(
            "Disabling /v1/ingest: secure auth mode required (set RAG_AUTH_MODE to 'api_key' or 'jwt')."
        )

        @app.post("/v1/ingest", response_model=IngestResponse)
        async def ingest_disabled(
            _request: IngestRequest,
            _background_tasks: BackgroundTasks,
        ) -> IngestResponse:
            """Reject ingest attempts when API auth is not enforced."""

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Ingest disabled until RAG_AUTH_MODE is set to 'api_key' or 'jwt'",
            )

    # ========================================================================
    # Metrics Endpoint (Placeholder)
    # ========================================================================

    @app.get("/v1/metrics")
    async def get_metrics(
        _: Dict[str, Any] = Depends(validate_request_credentials),
    ) -> Dict[str, Any]:
        """Get system metrics (placeholder for Prometheus integration).

        Returns:
            Dictionary with metrics (JSON for easy parsing)
        """
        with app.state.index_lock:
            chunks = app.state.chunks
            index_ready = app.state.index_ready

        return {
            "timestamp": datetime.now().isoformat(),
            "index_ready": index_ready,
            "chunks_loaded": len(chunks) if chunks else 0,
        }

    return app


# ============================================================================
# Standalone Server
# ============================================================================


app = create_app()


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    workers: int = 4,
    log_level: str = "info",
) -> None:
    """Run the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to
        workers: Number of worker processes
        log_level: Logging level
    """
    import uvicorn

    uvicorn.run(
        "clockify_rag.api:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=False,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
