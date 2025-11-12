import json

import clockify_rag.cli as cli_module
from clockify_rag.cli import chat_repl


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

    expected_citations = [1, 2, 3]
    expected_tokens = 987

    result_payload = {
        "answer": "Mocked answer",
        "confidence": 0.82,
        "selected_chunks": expected_citations,
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
