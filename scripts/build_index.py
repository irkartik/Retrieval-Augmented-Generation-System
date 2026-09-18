"""Ingestion -> chunk -> embed -> Chroma index, end to end (no API key needed; local embeddings).

Usage: python -m scripts.build_index
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunking import chunk_paragraphs
from rag.config import load_config, resolve_path
from rag.embeddings import get_embedder
from rag.ingestion import iter_paragraphs, load_squad, select_topics
from rag.vectorstore import build_vectorstore


def main():
    cfg = load_config()
    corpus_cfg = cfg["corpus"]

    squad_path = resolve_path(cfg, corpus_cfg["path"])
    squad_data = load_squad(squad_path)
    selected = select_topics(squad_data, corpus_cfg["topics"])
    paragraphs = iter_paragraphs(selected)
    print(f"Loaded {len(paragraphs)} paragraphs across {len(corpus_cfg['topics'])} topics")

    chunks = chunk_paragraphs(
        paragraphs,
        chunk_size=cfg["chunking"]["chunk_size"],
        chunk_overlap=cfg["chunking"]["chunk_overlap"],
    )
    print(f"Split into {len(chunks)} chunks (chunk_size={cfg['chunking']['chunk_size']} tokens)")

    embedder = get_embedder(cfg)
    store = build_vectorstore(chunks, embedder, cfg)
    print(f"Indexed {len(chunks)} chunks into Chroma collection '{cfg['vectorstore']['collection_name']}'")
    return store


if __name__ == "__main__":
    main()
