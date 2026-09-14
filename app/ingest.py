"""Builds (or loads) the Chroma vector store from the PDF corpus in data/raw/.

Cloud Run instances have ephemeral local disk, so the persisted store can't simply
ship baked into the image and survive across deploys/instances. Given this project's
minimal scope, the store is instead built lazily: on first use (or explicitly via
`python -m app.ingest`), the PDFs are loaded, split, embedded, and persisted to
CHROMA_PERSIST_DIR; subsequent calls in the same container just load what's there.
See docs/BUILD_LOG.md for the full tradeoff discussion.
"""

import pathlib

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Config


def _load_documents(raw_dir: str):
    pdf_paths = sorted(pathlib.Path(raw_dir).glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {raw_dir}. Run scripts/download_corpus.py first."
        )
    documents = []
    for pdf_path in pdf_paths:
        documents.extend(PyPDFLoader(str(pdf_path)).load())
    return documents


def _embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=Config.EMBEDDING_MODEL, google_api_key=Config.GOOGLE_API_KEY
    )


def build_vector_store() -> Chroma:
    documents = _load_documents(Config.RAW_DOCS_DIR)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    return Chroma.from_documents(
        chunks, embedding=_embeddings(), persist_directory=Config.CHROMA_PERSIST_DIR
    )


def load_or_build_vector_store() -> Chroma:
    persist_dir = pathlib.Path(Config.CHROMA_PERSIST_DIR)
    if persist_dir.exists() and any(persist_dir.iterdir()):
        return Chroma(
            persist_directory=Config.CHROMA_PERSIST_DIR, embedding_function=_embeddings()
        )
    return build_vector_store()


if __name__ == "__main__":
    build_vector_store()
    print(f"vector store persisted to {Config.CHROMA_PERSIST_DIR}")
