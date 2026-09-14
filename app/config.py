import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "models/gemini-embedding-001")
    CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "/tmp/chroma_db")
    RAW_DOCS_DIR = os.environ.get("RAW_DOCS_DIR", "data/raw")
    RETRIEVAL_K = int(os.environ.get("RETRIEVAL_K", "4"))
    CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))
