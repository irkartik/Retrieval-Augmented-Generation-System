"""Chroma vector store build/load, config-driven."""
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from rag.config import resolve_path


def build_vectorstore(chunks: list[dict], embedder, cfg: dict) -> Chroma:
    docs = [Document(page_content=c["text"], metadata=c["metadata"]) for c in chunks]
    ids = [c["id"] for c in chunks]  # stable IDs => re-running build_index upserts, not duplicates
    persist_path = str(resolve_path(cfg, cfg["vectorstore"]["persist_path"]))
    store = Chroma.from_documents(
        documents=docs,
        embedding=embedder,
        ids=ids,
        collection_name=cfg["vectorstore"]["collection_name"],
        persist_directory=persist_path,
        collection_metadata={"hnsw:space": "cosine"},  # MiniLM is trained for cosine; keeps scores in [0,1]
    )
    return store


def load_vectorstore(embedder, cfg: dict) -> Chroma:
    persist_path = str(resolve_path(cfg, cfg["vectorstore"]["persist_path"]))
    return Chroma(
        collection_name=cfg["vectorstore"]["collection_name"],
        embedding_function=embedder,
        persist_directory=persist_path,
    )
