import types

import pytest

import clockify_rag.cli as cli


class DummyArgs(types.SimpleNamespace):
    pass


def test_determinism_uses_dict_response(monkeypatch, capsys):
    call_count = {"value": 0}

    def fake_post(url, payload, retries=0, timeout=None, **_):
        call_count["value"] += 1
        return {"response": "Clockify is a time tracking tool."}

    monkeypatch.setattr(cli, "http_post_with_retries", fake_post)
    monkeypatch.setattr(cli, "load_index", lambda: ([], [], None, None))
    monkeypatch.setattr(cli.os.path, "exists", lambda _path: True)

    args = DummyArgs(
        det_check=True,
        retries=0,
        topk=1,
        pack=1,
        threshold=0.5,
        rerank=False,
        debug=False,
        seed=42,
        num_ctx=128,
        num_predict=32,
        json=False,
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.handle_chat_command(args)

    assert excinfo.value.code == 0
    assert call_count["value"] == 2
    out = capsys.readouterr().out
    assert "deterministic=true" in out
