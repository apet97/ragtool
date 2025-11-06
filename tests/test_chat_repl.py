import json

import clockify_support_cli_final as cli


def test_chat_repl_json_output(monkeypatch, capsys):
    # Ensure environment setup routines are no-ops for the test
    monkeypatch.setattr(cli, "_log_config_summary", lambda **_: None)
    monkeypatch.setattr(cli, "warmup_on_startup", lambda: None)
    monkeypatch.setattr(cli.os.path, "exists", lambda path: True)
    monkeypatch.setattr(cli, "build", lambda *_, **__: None)

    # Provide deterministic artifacts and retrieval response
    chunks = {"chunk-1": {"id": "chunk-1", "text": "Citation text"}}
    monkeypatch.setattr(cli, "load_index", lambda: (chunks, object(), object(), object()))

    expected_citations = [{"id": "chunk-1", "text": "Citation text"}]
    expected_tokens = 987

    def fake_answer_once(*args, **kwargs):
        return "Mocked answer", {
            "selected": expected_citations,
            "used_tokens": expected_tokens,
        }

    monkeypatch.setattr(cli, "answer_once", fake_answer_once)

    # Simulate a single question followed by EOF to exit the REPL
    inputs = iter(["What is Clockify?"])

    def fake_input(_prompt=""):
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)

    cli.chat_repl(use_json=True)

    captured = capsys.readouterr().out
    json_start = captured.index("{")
    json_payload = captured[json_start:]
    output = json.loads(json_payload)

    assert output["debug"]["meta"]["used_tokens"] == expected_tokens
    assert output["citations"] == expected_citations
