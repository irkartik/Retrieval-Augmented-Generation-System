from rag.retrievers.base import Retriever, RetrievedChunk
from rag.retrievers.closed_book import ClosedBookRetriever
from rag.retrievers.dense import DenseRetriever

_NOT_YET_IMPLEMENTED = {
    "bm25": "the Final comparative study — see Dissertation_End_to_End_Plan.md",
    "hybrid": "the Final comparative study — see Dissertation_End_to_End_Plan.md",
    "rerank": "the Final comparative study — see Dissertation_End_to_End_Plan.md",
}


def get_retriever(strategy: str, cfg: dict, vectorstore=None) -> Retriever:
    if strategy == "dense":
        if vectorstore is None:
            raise ValueError("dense retriever requires a vectorstore")
        return DenseRetriever(vectorstore)
    if strategy == "closed_book":
        return ClosedBookRetriever()
    if strategy in _NOT_YET_IMPLEMENTED:
        raise NotImplementedError(
            f"retriever strategy '{strategy}' is not implemented yet — planned for {_NOT_YET_IMPLEMENTED[strategy]}"
        )
    raise ValueError(f"unknown retriever strategy: {strategy}")
