"""Tests for the Flask API. The RAG chain is mocked out via app.main.get_rag_chain."""

from unittest.mock import patch

import pytest

from app.main import create_app


@pytest.fixture
def client():
    app = create_app()
    app.testing = True
    return app.test_client()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_ask_requires_question(client):
    response = client.post("/ask", json={})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_ask_rejects_blank_question(client):
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400


def test_ask_returns_answer(client):
    fake_chain = type(
        "FakeChain", (), {"ask": lambda self, q: {"answer": "42", "sources": ["doc.pdf"]}}
    )()
    with patch("app.main.get_rag_chain", return_value=fake_chain):
        response = client.post("/ask", json={"question": "What is the answer?"})
    assert response.status_code == 200
    assert response.get_json() == {"answer": "42", "sources": ["doc.pdf"]}


def test_ask_handles_chain_errors(client):
    def _raise(self, q):
        raise RuntimeError("boom")

    fake_chain = type("FakeChain", (), {"ask": _raise})()
    with patch("app.main.get_rag_chain", return_value=fake_chain):
        response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 500
