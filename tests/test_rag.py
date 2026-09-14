"""Tests for the RAG chain. Fully mocked — no network calls, no API key needed.

The retriever and LLM are swapped for RunnableLambda fakes so the real LCEL plumbing
(RunnableParallel, the prompt template, StrOutputParser) is exercised end to end.
"""

from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from app.rag import RagChain, _format_docs


def _fake_docs():
    return [
        Document(
            page_content="RAG combines retrieval with generation.",
            metadata={"source": "data/raw/retrieval_augmented_generation.pdf", "page": 1},
        ),
        Document(
            page_content="Attention is all you need.",
            metadata={"source": "data/raw/attention_is_all_you_need.pdf", "page": 3},
        ),
    ]


def test_format_docs_includes_source_and_page():
    formatted = _format_docs(_fake_docs())
    assert "retrieval_augmented_generation.pdf" in formatted
    assert "attention_is_all_you_need.pdf" in formatted
    assert "page 1" in formatted
    assert "page 3" in formatted


def test_rag_chain_returns_answer_and_sources():
    fake_retriever = RunnableLambda(lambda question: _fake_docs())
    fake_vector_store = type(
        "FakeVectorStore", (), {"as_retriever": lambda self, **kw: fake_retriever}
    )()
    fake_llm = RunnableLambda(lambda prompt_value: "RAG is retrieval-augmented generation.")

    with (
        patch("app.rag.load_or_build_vector_store", return_value=fake_vector_store),
        patch("app.rag.ChatGoogleGenerativeAI", return_value=fake_llm),
    ):
        chain = RagChain()
        result = chain.ask("What is RAG?")

    assert result["answer"] == "RAG is retrieval-augmented generation."
    assert result["sources"] == [
        "attention_is_all_you_need.pdf",
        "retrieval_augmented_generation.pdf",
    ]
