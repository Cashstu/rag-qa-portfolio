"""Flask application: RAG Q&A API. Entrypoint for gunicorn as `app.main:app`."""

import logging

from flask import Flask, jsonify, request

logger = logging.getLogger(__name__)

_rag_chain = None


def get_rag_chain():
    """Lazily builds the RagChain on first use, not at import time.

    Import-time construction would make `/health` (and even importing this module
    for tests) depend on having a valid GOOGLE_API_KEY and a built vector store.
    """
    global _rag_chain
    if _rag_chain is None:
        from app.rag import RagChain

        _rag_chain = RagChain()
    return _rag_chain


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.post("/ask")
    def ask():
        payload = request.get_json(silent=True) or {}
        question = (payload.get("question") or "").strip()
        if not question:
            return jsonify(error="'question' is required"), 400
        try:
            result = get_rag_chain().ask(question)
        except Exception:
            logger.exception("failed to answer question")
            return jsonify(error="failed to generate an answer"), 500
        return jsonify(result)

    return app


app = create_app()
