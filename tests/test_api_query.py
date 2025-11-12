from fastapi.testclient import TestClient

import clockify_rag.api as api_module


def test_api_query_returns_metadata(monkeypatch):
    """API should expose metadata from the new answer_once result schema."""

    monkeypatch.setattr(api_module, "ensure_index_ready", lambda retries=2: ([], [], {}, None))

    result_payload = {
        "answer": "Mocked answer",
        "confidence": 0.91,
        "selected_chunks": [3, 4, 5],
        "metadata": {"used_tokens": 128, "retrieval_count": 3},
        "routing": {"action": "self-serve"},
    }

    monkeypatch.setattr(api_module, "answer_once", lambda *_, **__: result_payload)

    app = api_module.create_app()

    with TestClient(app) as client:
        response = client.post(
            "/v1/query",
            json={"question": "How do I track time?"},
        )

        assert response.status_code == 200
        payload = response.json()

        assert payload["answer"] == result_payload["answer"]
        assert payload["sources"] == result_payload["selected_chunks"][:5]
        assert payload["metadata"]["used_tokens"] == result_payload["metadata"]["used_tokens"]
        assert payload["routing"] == result_payload["routing"]
