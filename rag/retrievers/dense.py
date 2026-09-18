from rag.retrievers.base import RetrievedChunk, Retriever


class DenseRetriever(Retriever):
    """Baseline semantic retriever: Chroma similarity search over sentence-transformer embeddings."""

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
        return [
            RetrievedChunk(
                text=doc.page_content,
                title=doc.metadata.get("title", ""),
                doc_id=doc.metadata.get("doc_id", ""),
                score=score,
            )
            for doc, score in results
        ]
