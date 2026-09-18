"""RAGAS faithfulness: are the answer's claims grounded in the retrieved context?

The anti-hallucination signal (master plan §5). Does NOT use the gold answer — it
compares the answer against the retrieved passages only, which is what makes it
orthogonal to EM/F1 correctness and lets us build the 2x2 matrix.

Wired to the same LLM the run uses (config `evaluation.llm_judge_*`), kept fixed across
strategies for a fair comparison. Requires an API key at run time, like all LLM steps.
"""
from rag.retrievers.base import RetrievedChunk


def build_faithfulness_metric(cfg: dict):
    """Construct a ragas Faithfulness metric backed by the configured judge LLM.

    Imports are local so the rest of the harness (EM/F1, retrieval) works without ragas
    or an API key present.
    """
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness

    provider = cfg["evaluation"]["llm_judge_provider"]
    model = cfg["evaluation"]["llm_judge_model"]

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        judge = ChatOpenAI(model=model, temperature=0.0)
    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "Anthropic judge needs langchain-anthropic — run `pip install langchain-anthropic`"
            ) from exc

        judge = ChatAnthropic(model=model, temperature=0.0)
    else:
        raise ValueError(f"unknown llm_judge_provider: {provider}")

    return Faithfulness(llm=LangchainLLMWrapper(judge))


def score_faithfulness(metric, question: str, answer: str, chunks: list[RetrievedChunk]) -> float:
    """Return a faithfulness score in [0, 1] for one (question, answer, retrieved-context) triple."""
    from ragas.dataset_schema import SingleTurnSample

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=[c.text for c in chunks],
    )
    return float(metric.single_turn_score(sample))
