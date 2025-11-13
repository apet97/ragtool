import json

import pytest
from typer.testing import CliRunner

import clockify_rag.cli_modern as cli_modern


@pytest.fixture
def cli_runner():
    return CliRunner()


def test_query_command_surfaces_metadata(monkeypatch, cli_runner):
    """Ensure Typer query command consumes new answer_once schema."""

    monkeypatch.setattr(cli_modern, "ensure_index_ready", lambda retries=2: ([], [], {}, None))

    result_payload = {
        "answer": "Mocked answer",
        "confidence": 0.77,
        "selected_chunks": [10, 20],
        "selected_chunk_ids": ["doc-10", "doc-20"],
        "metadata": {"used_tokens": 42},
    }

    monkeypatch.setattr(cli_modern, "answer_once", lambda *_, **__: result_payload)

    response = cli_runner.invoke(
        cli_modern.app,
        [
            "query",
            "How do I track time?",
            "--json",
        ],
    )

    assert response.exit_code == 0
    payload = json.loads("{" + response.stdout.split("{", 1)[1])

    assert payload["answer"] == result_payload["answer"]
    assert payload["sources"] == result_payload["selected_chunk_ids"]
    assert payload["metadata"]["used_tokens"] == result_payload["metadata"]["used_tokens"]
