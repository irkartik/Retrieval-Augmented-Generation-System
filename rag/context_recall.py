"""Deterministic context recall (retrieval hit-rate@k) for the SQuAD slice.

Where correctness (EM/F1) and faithfulness judge the *answer*, context recall judges the
*retriever*: did the top-k retrieved chunks include the gold SQuAD source paragraph the
question was written against? Because SQuAD hands us that gold paragraph, this is computed
by exact paragraph identity (doc_id) — no LLM judge, no API key, fully reproducible.

A question is a "hit" if any retrieved chunk shares the doc_id of its gold paragraph; the
mean over questions is hit-rate@k. It is meaningless for the closed-book baseline (no
retrieved context) — scored as None there and excluded from the mean, never counted as a miss.
"""
from rag.config import resolve_path
from rag.ingestion import iter_paragraphs, load_squad, select_topics


def build_context_to_doc_id(cfg: dict) -> dict[str, str]:
    """Map each SQuAD paragraph's text -> its doc_id, over the locked topic slice.

    Ingestion assigns doc_id deterministically (`{title}::{paragraph_index}`) and the frozen
    ground-truth stores the paragraph verbatim as `gold_context`, so this recovers the gold
    paragraph id for every question without storing it twice.
    """
    squad_path = resolve_path(cfg, cfg["corpus"]["path"])
    paragraphs = iter_paragraphs(select_topics(load_squad(squad_path), cfg["corpus"]["topics"]))
    mapping: dict[str, str] = {}
    for para in paragraphs:
        mapping.setdefault(para.context, para.doc_id)  # first wins on the rare duplicate context
    return mapping


def gold_doc_id_for(qa: dict, context_to_doc_id: dict[str, str]) -> str | None:
    """The gold paragraph's doc_id for one ground-truth question, or None if unresolved."""
    return context_to_doc_id.get(qa.get("gold_context"))


def context_recall_hit(gold_doc_id: str | None, retrieved_doc_ids) -> float | None:
    """1.0 if the gold paragraph is among the retrieved chunks, 0.0 if not, None if unscoreable.

    Unscoreable = no retrieved context (closed-book) or an unresolved gold paragraph; returned
    as None so such questions are excluded from the mean rather than penalised as misses.
    """
    if not retrieved_doc_ids or gold_doc_id is None:
        return None
    return 1.0 if gold_doc_id in set(retrieved_doc_ids) else 0.0
