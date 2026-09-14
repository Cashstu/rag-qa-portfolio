"""Builds the retrieval-augmented generation chain (LCEL) used by POST /ask.

Retrieves once per question and fans that single retrieval out to both the answer
generation and the reported sources, rather than retrieving twice.
"""

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import Config
from app.ingest import load_or_build_vector_store

PROMPT = ChatPromptTemplate.from_template(
    "You are a research assistant. Answer the question using ONLY the context "
    "below. If the context does not contain the answer, say you don't know — "
    "do not make anything up.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def _format_docs(docs) -> str:
    return "\n\n".join(
        f"[{os.path.basename(doc.metadata.get('source', 'unknown'))}, "
        f"page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )


def _sources(docs) -> list[str]:
    return sorted({os.path.basename(doc.metadata.get("source", "unknown")) for doc in docs})


class RagChain:
    def __init__(self):
        vector_store = load_or_build_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": Config.RETRIEVAL_K})
        llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL, google_api_key=Config.GOOGLE_API_KEY
        )

        answer_chain = (
            {
                "context": lambda x: _format_docs(x["docs"]),
                "question": lambda x: x["question"],
            }
            | PROMPT
            | llm
            | StrOutputParser()
        )

        self.chain = RunnableParallel(
            docs=retriever, question=RunnablePassthrough()
        ) | RunnableParallel(
            answer=answer_chain,
            sources=lambda x: _sources(x["docs"]),
        )

    def ask(self, question: str) -> dict:
        return self.chain.invoke(question)
