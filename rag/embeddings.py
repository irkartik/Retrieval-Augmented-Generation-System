"""Local, free embedding model (sentence-transformers) — no API key needed."""
from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedder(cfg: dict) -> HuggingFaceEmbeddings:
    model_name = cfg["embedding"]["model"]
    return HuggingFaceEmbeddings(model_name=f"sentence-transformers/{model_name}")
