import json

import clockify_rag.cli as cli_module
from clockify_rag.cli import chat_repl
from clockify_rag.exceptions import ValidationError


def test_chat_repl_json_output(monkeypatch, capsys):
    """Ensure REPL surfaces new answer_once metadata in JSON mode."""

    # Stub expensive startup routines
    monkeypatch.setattr(cli_module, "_log_config_summary", lambda **_: None)
    monkeypatch.setattr(cli_module, "warmup_on_startup", lambda: None)

    class DummyCache:
        def __init__(self):
            self.cache = {}

        def load(self):
            return None

        def save(self):
            return None

    monkeypatch.setattr(cli_module, "get_query_cache", lambda: DummyCache())
    monkeypatch.setattr(cli_module, "get_precomputed_cache", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli_module, "ensure_index_ready", lambda **_: ([], [], {}, None))

    expected_citations = ["doc-1", "doc-2", "doc-3"]
    expected_tokens = 987

    result_payload = {
        "answer": "Mocked answer",
        "confidence": 0.82,
        "selected_chunks": [0, 1, 2],
        "selected_chunk_ids": expected_citations,
        "metadata": {"used_tokens": expected_tokens, "retrieval_count": len(expected_citations)},
    }

    monkeypatch.setattr(cli_module, "answer_once", lambda *_, **__: result_payload)

    inputs = iter(["What is Clockify?", ":exit"])

    def fake_input(_prompt: str = ""):
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)

    chat_repl(use_json=True)

    captured = capsys.readouterr().out
    json_start = captured.index("{")
    output = json.loads(captured[json_start:])

    assert output["used_tokens"] == expected_tokens
    assert output["citations"] == expected_citations


def test_chat_repl_handles_validation_errors(monkeypatch, capsys):
    """REPL should continue after validation errors and show messages."""

    monkeypatch.setattr(cli_module, "_log_config_summary", lambda **_: None)
    monkeypatch.setattr(cli_module, "warmup_on_startup", lambda: None)

    class DummyCache:
        def __init__(self):
            self.cache = {}

        def load(self):
            return None

        def save(self):
            return None

    monkeypatch.setattr(cli_module, "get_query_cache", lambda: DummyCache())
    monkeypatch.setattr(cli_module, "get_precomputed_cache", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli_module, "ensure_index_ready", lambda **_: ([], [], {}, None))

    result_payload = {
        "answer": "Valid answer",
        "selected_chunks": [7],
        "metadata": {},
    }

    error_messages = [
        "Query cannot be empty",
        "Query too long (12001 chars). Maximum allowed: 12000 chars. Set MAX_QUERY_LENGTH env var to override.",
    ]

    call_count = {"value": 0}

    def fake_answer(*_args, **_kwargs):
        idx = call_count["value"]
        call_count["value"] += 1
        if idx < len(error_messages):
            raise ValidationError(error_messages[idx])
        return result_payload

    monkeypatch.setattr(cli_module, "answer_once", fake_answer)

    inputs = iter(["first", "second", "valid", ":exit"])

    def fake_input(_prompt: str = ""):
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)

    chat_repl()

    captured = capsys.readouterr().out
    assert "Valid answer" in captured
    for message in error_messages:
        assert message in captured
