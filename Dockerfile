# Multi-architecture Dockerfile for Clockify RAG
# Supports: linux/amd64, linux/arm64 (Apple Silicon via Docker Desktop)
#
# Build with:
#   docker build -t clockify-rag:latest .
#
# Build for multiple architectures:
#   docker buildx build --platform linux/amd64,linux/arm64 -t clockify-rag:latest .

# Stage 1: Builder
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast dependency manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy dependency files (layer caching optimization)
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY clockify_rag/__init__.py clockify_rag/__init__.py

# Install Python dependencies (production only, no dev extras)
RUN /root/.cargo/bin/uv pip install --python /usr/local/bin/python3.11 \
    --target /app/venv \
    --compile-bytecode \
    .
# Note: Using '.' instead of '-e .' to avoid dev dependencies
# This installs only the dependencies listed in pyproject.toml [project.dependencies]

# Stage 2: Runtime
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/venv/bin:$PATH"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy venv from builder
COPY --from=builder /app/venv /app/venv

# Create non-root user for security
RUN groupadd -r raguser && useradd -r -g raguser raguser

# Copy application code
COPY --chown=raguser:raguser . .

# Create necessary directories with proper permissions
RUN mkdir -p var/{index,logs,reports,backups} /app/.cache && \
    chown -R raguser:raguser var /app/.cache && \
    chmod 755 var /app/.cache && \
    chmod 755 var/{index,logs,reports,backups}

# Switch to non-root user
USER raguser

# Ensure writable directories exist and have correct permissions
RUN touch /app/var/logs/.keep && \
    touch /app/var/index/.keep

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# Expose port for API server
EXPOSE 8000

# Default command: run API server with graceful shutdown (30s timeout)
CMD ["python", "-m", "uvicorn", "clockify_rag.api:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "30"]

# Alternative commands:
# - Interactive CLI: docker run -it clockify-rag:latest python -m clockify_rag.cli_modern chat
# - Build index: docker run -v $(pwd)/knowledge_full.md:/app/knowledge_full.md clockify-rag:latest python -m clockify_rag.cli_modern ingest
# - Single query: docker run clockify-rag:latest python -m clockify_rag.cli_modern query "Your question"
