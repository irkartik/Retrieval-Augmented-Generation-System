"""Ask one question against the built index and get a grounded, cited answer.

Requires scripts/build_index.py to have been run first, and an API key for
cfg['generation']['provider'] set in .env.

Usage: python -m scripts.run_query "Who did Nikola Tesla work for?"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from rag.config import load_config
from rag.embeddings import get_embedder
from rag.generation import generate_answer
from rag.retrievers import get_retriever
from rag.vectorstore import load_vectorstore


def main():
    load_dotenv()
    if len(sys.argv) < 2:
        print('Usage: python -m scripts.run_query "your question"')
        sys.exit(1)
    question = sys.argv[1]

    cfg = load_config()
    embedder = get_embedder(cfg)
    store = load_vectorstore(embedder, cfg)
    retriever = get_retriever(cfg["retriever"]["strategy"], cfg, vectorstore=store)

    chunks = retriever.retrieve(question, top_k=cfg["retriever"]["top_k"])
    result = generate_answer(question, chunks, cfg)

    print(f"\nQ: {question}\n")
    print(f"A: {result['answer']}\n")
    print("Citations:")
    for c in result["citations"]:
        print(f"  - {c['title']} ({c['doc_id']})")


if __name__ == "__main__":
    main()
