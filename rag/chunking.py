"""Token-based chunking, parameterised by chunk_size/chunk_overlap (config-driven)."""
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.ingestion import Paragraph


def chunk_paragraphs(paragraphs: list[Paragraph], chunk_size: int, chunk_overlap: int) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = []
    for para in paragraphs:
        pieces = splitter.split_text(para.context)
        for i, piece in enumerate(pieces):
            chunks.append(
                {
                    "id": f"{para.doc_id}::chunk{i}",
                    "text": piece,
                    "metadata": {"title": para.title, "doc_id": para.doc_id},
                }
            )
    return chunks
