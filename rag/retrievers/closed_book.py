from rag.retrievers.base import RetrievedChunk, Retriever


class ClosedBookRetriever(Retriever):
    """No-retrieval baseline (examiner-feedback fix): the LLM answers from parametric memory only.

    Used to prove correctness alone doesn't track retrieval quality — see
    Examiner_Feedback_Response.md §2.
    """

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        return []
