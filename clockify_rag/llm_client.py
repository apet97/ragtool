"""Remote-first LLM client factory using LangChain ChatOllama.

This module provides a clean, production-ready interface for LLM calls via
the corporate Ollama instance. Designed for VPN environments with:

- Non-streaming generation (VPN stability)
- Configurable timeouts (default 120s)
- Transparent fallback model selection
- VPN-safe error handling (no indefinite hangs)
"""

import logging
from typing import Optional

import httpx

# Use langchain-ollama (newer package, better maintained)
try:
    from langchain_ollama import ChatOllama
except ImportError:
    # Fallback to langchain-community for older installations
    from langchain_community.chat_models import ChatOllama  # type: ignore

from .config import RAG_OLLAMA_URL, LLM_MODEL, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)


def get_llm_client(temperature: float = 0.0) -> ChatOllama:
    """Create and return a ChatOllama client for remote generation.

    This is the single source of truth for all LLM calls in the system.
    Uses remote-first design: connects to corporate Ollama over VPN with
    non-streaming mode for stability and explicit timeout controls.

    Args:
        temperature: Sampling temperature (0.0-1.0; 0.0 = deterministic)

    Returns:
        ChatOllama instance configured for remote generation with:
        - Non-streaming (VPN safe, no infinite hangs)
        - Timeout enforcement (120s default via OLLAMA_TIMEOUT)
        - Selected model (with automatic fallback applied at config import time)
        - Base URL: RAG_OLLAMA_URL (corporate Ollama instance)

    Usage:
        ```python
        from clockify_rag.llm_client import get_llm_client

        llm = get_llm_client(temperature=0.0)
        response = llm.invoke("What is 2+2?")
        print(response.content)
        ```

    Notes:
        - Model selection happens at config import time (_select_best_model)
        - If primary model unavailable, falls back to RAG_CHAT_FALLBACK_MODEL
        - If Ollama unreachable at startup, uses primary anyway (assumes VPN will reconnect)
        - All calls timeout after OLLAMA_TIMEOUT seconds (default 120s, configurable)
    """
    logger.debug(
        f"Creating ChatOllama client: model={LLM_MODEL}, "
        f"base_url={RAG_OLLAMA_URL}, timeout={OLLAMA_TIMEOUT}s, streaming=False"
    )

    # Use httpx.Client with explicit timeout for version-robust timeout handling
    # Some langchain versions don't accept timeout= kwarg directly on ChatOllama
    http_client = httpx.Client(timeout=OLLAMA_TIMEOUT)

    return ChatOllama(
        base_url=RAG_OLLAMA_URL,
        model=LLM_MODEL,
        temperature=temperature,
        # VPN safety: never stream over flaky corporate networks
        # Non-streaming ensures predictable request completion time
        streaming=False,
        # Pass httpx client with timeout configured
        # This is the most version-robust way to set timeouts in langchain
        client=http_client,
    )


def get_llm_client_async(temperature: float = 0.0) -> ChatOllama:
    """Create an async-capable ChatOllama client (experimental).

    Note: LangChain's ChatOllama doesn't have native async support yet.
    This function returns the same as get_llm_client() but documents the
    interface for future async work.

    Args:
        temperature: Sampling temperature (0.0-1.0)

    Returns:
        ChatOllama instance (currently synchronous)
    """
    # TODO: Switch to async client once langchain-community adds native async support
    return get_llm_client(temperature)
