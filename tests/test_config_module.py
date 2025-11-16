import importlib.util
from pathlib import Path
import uuid

import pytest
from typing import Optional


CONFIG_PATH = Path(__file__).resolve().parents[1] / "clockify_rag" / "config.py"
CONFIG_ENV_KEYS = [
    "RAG_OLLAMA_URL",
    "RAG_LLM_CLIENT",
    "OLLAMA_URL",
    "RAG_CHAT_MODEL",
    "GEN_MODEL",
    "CHAT_MODEL",
    "RAG_EMBED_MODEL",
    "EMB_MODEL",
    "EMBED_MODEL",
    "EMB_BACKEND",
    "DEFAULT_TOP_K",
    "DEFAULT_PACK_TOP",
    "DEFAULT_THRESHOLD",
    "DEFAULT_NUM_CTX",
    "DEFAULT_NUM_PREDICT",
    "DEFAULT_RETRIES",
    "CHUNK_CHARS",
    "CHUNK_OVERLAP",
    "CTX_BUDGET",
    "ANN",
    "ANN_NLIST",
    "ANN_NPROBE",
    "FAISS_CANDIDATE_MULTIPLIER",
    "ANN_CANDIDATE_MIN",
    "BM25_K1",
    "BM25_B",
    "EMB_CONNECT_TIMEOUT",
    "EMB_READ_TIMEOUT",
    "CHAT_CONNECT_TIMEOUT",
    "CHAT_READ_TIMEOUT",
    "RERANK_READ_TIMEOUT",
    "EMB_MAX_WORKERS",
    "EMB_BATCH_SIZE",
    "MMR_LAMBDA",
    "USE_INTENT_CLASSIFICATION",
    "MAX_QUERY_LENGTH",
    "RAG_LOG_FILE",
    "RAG_LOG_INCLUDE_ANSWER",
    "RAG_LOG_INCLUDE_CHUNKS",
    "RAG_LOG_ANSWER_PLACEHOLDER",
    "RAG_STRICT_CITATIONS",
    "CACHE_MAXSIZE",
    "CACHE_TTL",
    "RATE_LIMIT_REQUESTS",
    "RATE_LIMIT_WINDOW",
    "WARMUP",
    "NLTK_AUTO_DOWNLOAD",
    "CLOCKIFY_QUERY_EXPANSIONS",
    "MAX_QUERY_EXPANSION_FILE_SIZE",
    "API_AUTH_MODE",
    "API_ALLOWED_KEYS",
    "API_KEY_HEADER",
    "ALLOW_PROXIES",
    "USE_PROXY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "BUILD_LOCK_TTL_SEC",
    "FAQ_CACHE_ENABLED",
    "FAQ_CACHE_PATH",
]


def _load_config_module(monkeypatch: pytest.MonkeyPatch, overrides: Optional[dict[str, str]] = None):
    """Load a fresh instance of the config module with supplied env overrides."""
    overrides = overrides or {}
    for key in CONFIG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    module_name = f"clockify_rag.config_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, CONFIG_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def test_config_defaults_match_expected(monkeypatch: pytest.MonkeyPatch):
    cfg = _load_config_module(monkeypatch)
    assert cfg.RAG_OLLAMA_URL == "http://10.127.0.192:11434"
    assert cfg.RAG_CHAT_MODEL == "qwen2.5:32b"
    assert cfg.RAG_EMBED_MODEL == "nomic-embed-text:latest"
    assert cfg.DEFAULT_TOP_K == 15
    assert cfg.DEFAULT_NUM_PREDICT == 512
    assert cfg.FAISS_CANDIDATE_MULTIPLIER == 3
    assert cfg.ANN_CANDIDATE_MIN == 200


def test_config_env_overrides_take_precedence(monkeypatch: pytest.MonkeyPatch):
    cfg = _load_config_module(
        monkeypatch,
        {
            "RAG_OLLAMA_URL": "http://example.com:11434",
            "RAG_CHAT_MODEL": "gpt-oss:20b",
            "RAG_EMBED_MODEL": "nomic-embed-text:latest",
            "DEFAULT_TOP_K": "25",
            "DEFAULT_THRESHOLD": "0.4",
            "DEFAULT_NUM_PREDICT": "1024",
            "FAISS_CANDIDATE_MULTIPLIER": "4",
            "ANN_CANDIDATE_MIN": "50",
        },
    )
    assert cfg.RAG_OLLAMA_URL == "http://example.com:11434"
    assert cfg.RAG_CHAT_MODEL == "gpt-oss:20b"
    assert cfg.DEFAULT_TOP_K == 25
    assert cfg.DEFAULT_THRESHOLD == 0.4
    assert cfg.DEFAULT_NUM_PREDICT == 1024
    assert cfg.FAISS_CANDIDATE_MULTIPLIER == 4
    assert cfg.ANN_CANDIDATE_MIN == 50


def test_config_supports_legacy_env_aliases(monkeypatch: pytest.MonkeyPatch):
    cfg = _load_config_module(
        monkeypatch,
        {
            "OLLAMA_URL": "http://legacy-host:11434",
            "GEN_MODEL": "legacy-gen",
            "EMB_MODEL": "legacy-emb",
        },
    )
    assert cfg.RAG_OLLAMA_URL == "http://legacy-host:11434"
    assert cfg.RAG_CHAT_MODEL == "legacy-gen"
    assert cfg.RAG_EMBED_MODEL == "legacy-emb"


def test_invalid_numeric_values_are_clamped(monkeypatch: pytest.MonkeyPatch):
    cfg = _load_config_module(
        monkeypatch,
        {
            "DEFAULT_TOP_K": "-5",  # below min
            "BM25_K1": "-10",  # below min
            "FAISS_CANDIDATE_MULTIPLIER": "999",  # above max
            "DEFAULT_THRESHOLD": "not-a-number",  # invalid float
        },
    )
    assert cfg.DEFAULT_TOP_K == 1  # clamped
    assert cfg.BM25_K1 == 0.1  # clamped minimum
    assert cfg.FAISS_CANDIDATE_MULTIPLIER == 10  # clamped maximum
    assert cfg.DEFAULT_THRESHOLD == 0.25  # fallback to default on parse failure


def test_llm_client_helper_tracks_runtime_env(monkeypatch: pytest.MonkeyPatch):
    cfg = _load_config_module(monkeypatch)
    # With env unset we fall back to provided default
    assert cfg.get_llm_client_mode(default="mock") == "mock"

    # Changing the environment after import updates the helper on next call
    monkeypatch.setenv("RAG_LLM_CLIENT", "Test")
    assert cfg.get_llm_client_mode() == "test"

    settings = cfg.current_llm_settings(default_client_mode="mock")
    assert settings.base_url == cfg.RAG_OLLAMA_URL
    assert settings.client_mode == "test"


def test_proxy_toggle_honors_legacy_alias(monkeypatch: pytest.MonkeyPatch):
    cfg = _load_config_module(monkeypatch, {"USE_PROXY": "1"})
    assert cfg.ALLOW_PROXIES is True

    cfg = _load_config_module(monkeypatch, {"ALLOW_PROXIES": "0"})
    assert cfg.ALLOW_PROXIES is False


def test_api_key_configuration_parses_csv(monkeypatch: pytest.MonkeyPatch):
    cfg = _load_config_module(
        monkeypatch,
        {
            "API_AUTH_MODE": "API_KEY",
            "API_ALLOWED_KEYS": "alpha,  beta , ,gamma",
            "API_KEY_HEADER": "X-Secret",
        },
    )
    assert cfg.API_AUTH_MODE == "api_key"
    assert cfg.API_ALLOWED_KEYS == frozenset({"alpha", "beta", "gamma"})
    assert cfg.API_KEY_HEADER == "X-Secret"


def test_blank_strings_are_treated_as_missing(monkeypatch: pytest.MonkeyPatch):
    cfg = _load_config_module(
        monkeypatch,
        {
            "CLOCKIFY_QUERY_EXPANSIONS": "   ",
            "HTTP_PROXY": "",
        },
    )
    assert cfg.CLOCKIFY_QUERY_EXPANSIONS is None
    assert cfg.HTTP_PROXY == ""
