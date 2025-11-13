import json
from argparse import Namespace

import pytest
from fastapi.testclient import TestClient

import clockify_rag.cli as cli
import clockify_rag.api as api_module
from clockify_rag import config


class _DummyLimiter:
    def allow_request(self, *_args, **_kwargs):
        return True

    def wait_time(self, *_args, **_kwargs):
        return 0


def _make_args(**overrides):
    defaults = dict(
        question="How do I track time?",
        topk=3,
        pack=2,
        threshold=0.25,
        rerank=False,
        seed=42,
        num_ctx=1024,
        num_predict=128,
        retries=0,
        json=False,
        debug=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_handle_ask_command_writes_log_when_enabled(tmp_path, monkeypatch):
    log_path = tmp_path / "queries.jsonl"
    monkeypatch.setattr(config, "QUERY_LOG_FILE", str(log_path))
    monkeypatch.setattr(cli, "QUERY_LOG_DISABLED", False, raising=False)

    fake_chunks = [{"id": "chunk-1", "title": "T", "section": "S"}]
    monkeypatch.setattr(cli, "ensure_index_ready", lambda retries=0: (fake_chunks, [], {}, None))
    result_payload = {
        "answer": "Mock answer",
        "selected_chunks": [0],
        "metadata": {"used_tokens": 12},
        "timing": {"total_ms": 12.5},
        "confidence": 0.88,
    }
    monkeypatch.setattr(cli, "answer_once", lambda *_, **__: result_payload)
    monkeypatch.setattr(cli, "get_rate_limiter", lambda: _DummyLimiter())
    monkeypatch.setattr("builtins.print", lambda *_args, **_kwargs: None)

    cli.handle_ask_command(_make_args())

    assert log_path.exists()
    payload = json.loads(log_path.read_text().strip())
    assert payload["query"] == "How do I track time?"
    assert payload["chunk_ids"] == ["chunk-1"]
    assert payload["metadata"]["confidence"] == pytest.approx(result_payload["confidence"])


def test_handle_ask_command_respects_no_log(tmp_path, monkeypatch):
    log_path = tmp_path / "queries.jsonl"
    log_path.write_text("sentinel")
    monkeypatch.setattr(config, "QUERY_LOG_FILE", str(log_path))
    monkeypatch.setattr(cli, "QUERY_LOG_DISABLED", True, raising=False)

    fake_chunks = [{"id": "chunk-1"}]
    monkeypatch.setattr(cli, "ensure_index_ready", lambda retries=0: (fake_chunks, [], {}, None))
    monkeypatch.setattr(cli, "answer_once", lambda *_, **__: {
        "answer": "Mock answer",
        "selected_chunks": [0],
        "metadata": {},
        "timing": {"total_ms": 3},
    })
    monkeypatch.setattr(cli, "get_rate_limiter", lambda: _DummyLimiter())
    monkeypatch.setattr("builtins.print", lambda *_args, **_kwargs: None)

    cli.handle_ask_command(_make_args())

    assert log_path.read_text() == "sentinel"


def test_api_privacy_mode_disables_logging(tmp_path, monkeypatch):
    log_path = tmp_path / "queries.jsonl"
    log_path.write_text("seed")
    monkeypatch.setattr(config, "QUERY_LOG_FILE", str(log_path))
    monkeypatch.setattr(api_module.config, "API_PRIVACY_MODE", True, raising=False)
    monkeypatch.setattr(api_module, "ensure_index_ready", lambda retries=2: ([{"id": "chunk-7"}], [], {}, None))
    result_payload = {
        "answer": "Mocked",
        "selected_chunks": [0],
        "metadata": {},
        "timing": {"total_ms": 9},
    }
    monkeypatch.setattr(api_module, "answer_once", lambda *_, **__: result_payload)

    app = api_module.create_app()
    with TestClient(app) as client:
        response = client.post("/v1/query", json={"question": "Hello?"})
        assert response.status_code == 200

    assert log_path.read_text() == "seed"
