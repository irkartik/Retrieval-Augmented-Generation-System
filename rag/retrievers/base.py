"""Common interface every retrieval strategy implements (dense/bm25/hybrid/rerank/closed_book)."""
from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    text: str
    title: str
    doc_id: str
    score: float


class Retriever:
    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        raise NotImplementedError
