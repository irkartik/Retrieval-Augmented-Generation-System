"""Turns retrieved chunks + a question into a grounded, cited answer (or a closed-book answer)."""
from rag.llm_providers import chat
from rag.retrievers.base import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a question-answering assistant. Answer the user's question as briefly and "
    "precisely as possible (a short phrase, not a full sentence, when the question expects "
    "a factual span). If context passages are provided, answer ONLY using those passages and "
    "cite the passage number(s) you used in square brackets, e.g. [1]. If no context passages "
    "are provided, answer from your own knowledge and do not fabricate citations."
)


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return f"Question: {question}\n\n(No context passages provided — closed-book mode.)"
    context_block = "\n\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(chunks))
    return f"Context passages:\n{context_block}\n\nQuestion: {question}"


def generate_answer(question: str, chunks: list[RetrievedChunk], cfg: dict) -> dict:
    user_prompt = build_user_prompt(question, chunks)
    result = chat(
        provider=cfg["generation"]["provider"],
        model=cfg["generation"]["model"],
        system=SYSTEM_PROMPT,
        user=user_prompt,
        temperature=cfg["generation"]["temperature"],
    )
    return {
        "answer": result.text,
        "citations": [{"title": c.title, "doc_id": c.doc_id} for c in chunks],
        "usage": {
            "model": result.model,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        },
    }
