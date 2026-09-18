"""Evaluation harness: correctness (EM/F1) + faithfulness (RAGAS) + latency + cost,
with the 2x2 faithfulness x correctness matrix per retrieval strategy.

Covers the Mid-Sem harness deliverable (Midsem_Plan.md §2 item 5) and the examiner-feedback
fix (Examiner_Feedback_Response.md §2): the closed-book baseline plus the correct-but-unfaithful
cell of the matrix. BM25/hybrid/rerank and the paired-stats layer are Final-report work.

Every LLM step (generation + faithfulness judge) needs an API key for
cfg['generation'] / cfg['evaluation'] set in .env.

Usage:
  python -m scripts.run_eval [--strategies dense,closed_book] [--limit 20] [--no-faithfulness]
"""
import argparse
import json
import re
import string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from rag.config import load_config, resolve_path
from rag.embeddings import get_embedder
from rag.faithfulness import build_faithfulness_metric, score_faithfulness
from rag.generation import generate_answer
from rag.pricing import estimate_cost_usd
from rag.retrievers import get_retriever
from rag.vectorstore import load_vectorstore

CITATION_RE = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")


def strip_citations(text: str) -> str:
    """Remove inline citation markers like [1] or [2, 3] before scoring.

    Retrieval arms are prompted to cite passages; the closed-book baseline is not.
    Scoring the raw text would penalise the cited (retrieval) answers only — an
    artefact that would bias the very comparison this study reports. Applied to the
    prediction only (gold answers never contain citations).
    """
    return CITATION_RE.sub(" ", text)


def normalize_answer(s: str) -> str:
    """SQuAD's official normalization: lowercase, strip punctuation/articles/extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match_score(prediction: str, gold_answers: list[str]) -> int:
    return int(any(normalize_answer(prediction) == normalize_answer(g) for g in gold_answers))


def f1_score(prediction: str, gold_answers: list[str]) -> float:
    def _f1(pred_tokens, gold_tokens):
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    pred_tokens = normalize_answer(prediction).split()
    return max(_f1(pred_tokens, normalize_answer(g).split()) for g in gold_answers)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def confusion_matrix(per_question: list[dict]) -> dict | None:
    """2x2 faithfulness x correctness matrix over questions that were faithfulness-scored.

    The dangerous cell is unfaithful_correct: the model got the right answer without
    grounding it in the retrieved context — i.e. answered from parametric memory
    (Examiner_Feedback_Response.md §2). Returns None if no question was scored.
    """
    scored = [r for r in per_question if r.get("faithful") is not None]
    if not scored:
        return None
    matrix = {"faithful_correct": 0, "faithful_incorrect": 0, "unfaithful_correct": 0, "unfaithful_incorrect": 0}
    for r in scored:
        bucket = ("faithful" if r["faithful"] else "unfaithful") + ("_correct" if r["correct"] else "_incorrect")
        matrix[bucket] += 1
    matrix["n_scored"] = len(scored)
    matrix["correct_but_unfaithful_rate"] = matrix["unfaithful_correct"] / len(scored)
    return matrix


def per_topic_breakdown(per_question: list[dict]) -> dict:
    by_topic = defaultdict(list)
    for r in per_question:
        by_topic[r["topic"]].append(r)
    return {
        topic: {"n": len(rows), "mean_em": _mean([r["em"] for r in rows]), "mean_f1": _mean([r["f1"] for r in rows])}
        for topic, rows in sorted(by_topic.items())
    }


def run_strategy(
    strategy: str, ground_truth: list[dict], cfg: dict, limit: int | None, faithfulness_enabled: bool
) -> dict:
    store = None
    if strategy != "closed_book":
        embedder = get_embedder(cfg)
        store = load_vectorstore(embedder, cfg)
    retriever = get_retriever(strategy, cfg, vectorstore=store)

    # Faithfulness is meaningless for closed-book (there is no retrieved context to be faithful to).
    score_faith = faithfulness_enabled and strategy != "closed_book"
    faith_metric = build_faithfulness_metric(cfg) if score_faith else None

    eval_cfg = cfg["evaluation"]
    faith_threshold = eval_cfg.get("faithfulness_threshold", 0.5)
    correct_threshold = eval_cfg.get("correctness_threshold", 0.5)  # token-F1 >= this counts as "correct"

    rows = ground_truth[:limit] if limit else ground_truth
    per_question = []
    for qa in rows:
        print(f"  [{strategy}] Q {len(per_question)+1}/{len(rows)}", flush=True)
        t0 = time.perf_counter()
        chunks = retriever.retrieve(qa["question"], top_k=cfg["retriever"]["top_k"])
        result = generate_answer(qa["question"], chunks, cfg)
        latency_s = time.perf_counter() - t0

        scored_pred = strip_citations(result["answer"])
        em = exact_match_score(scored_pred, qa["gold_answers"])
        f1 = f1_score(scored_pred, qa["gold_answers"])

        usage = result["usage"]
        cost_usd = estimate_cost_usd(usage["model"], usage["prompt_tokens"], usage["completion_tokens"])

        faith_score = None
        faithful = None
        if score_faith:
            faith_score = score_faithfulness(faith_metric, qa["question"], result["answer"], chunks)
            faithful = faith_score >= faith_threshold

        per_question.append(
            {
                "id": qa["id"],
                "topic": qa["topic"],
                "question": qa["question"],  # needed for post-hoc faithfulness scoring
                "em": em,
                "f1": f1,
                "correct": f1 >= correct_threshold,
                "faithfulness": faith_score,
                "faithful": faithful,
                "latency_s": latency_s,
                "cost_usd": cost_usd,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "prediction": result["answer"],  # raw answer (with citations) kept for auditing
                "retrieved_chunks": [{"text": c.text, "title": c.title, "doc_id": c.doc_id, "score": c.score} for c in chunks],  # needed for post-hoc faithfulness scoring
            }
        )

    n = len(per_question)
    faith_scores = [r["faithfulness"] for r in per_question if r["faithfulness"] is not None]
    costs = [r["cost_usd"] for r in per_question if r["cost_usd"] is not None]
    return {
        "strategy": strategy,
        "n": n,
        "mean_em": _mean([r["em"] for r in per_question]),
        "mean_f1": _mean([r["f1"] for r in per_question]),
        "mean_faithfulness": _mean(faith_scores) if faith_scores else None,
        "mean_latency_s": _mean([r["latency_s"] for r in per_question]),
        "total_cost_usd": sum(costs) if costs else None,
        "mean_cost_usd": _mean(costs) if costs else None,
        "confusion_matrix": confusion_matrix(per_question),
        "per_topic": per_topic_breakdown(per_question),
        "per_question": per_question,
    }


def _fmt(value, spec="{:.3f}"):
    return spec.format(value) if value is not None else "n/a"


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", default="dense,closed_book")
    parser.add_argument("--limit", type=int, default=None, help="pilot run size, e.g. 20")
    parser.add_argument(
        "--no-faithfulness",
        dest="faithfulness",
        action="store_false",
        help="skip the RAGAS faithfulness judge (faster/cheaper; EM/F1/latency/cost only)",
    )
    parser.set_defaults(faithfulness=True)
    args = parser.parse_args()

    cfg = load_config()
    gt_path = resolve_path(cfg, cfg["corpus"]["ground_truth_path"])
    with open(gt_path) as f:
        ground_truth = json.load(f)

    results = {}
    for strategy in args.strategies.split(","):
        print(f"Running strategy: {strategy} (n={args.limit or len(ground_truth)})")
        r = run_strategy(strategy, ground_truth, cfg, args.limit, args.faithfulness)
        results[strategy] = r
        print(
            f"  EM={_fmt(r['mean_em'])}  F1={_fmt(r['mean_f1'])}  "
            f"faithfulness={_fmt(r['mean_faithfulness'])}  "
            f"latency={_fmt(r['mean_latency_s'], '{:.2f}')}s  cost/q=${_fmt(r['mean_cost_usd'], '{:.5f}')}"
        )
        if r["confusion_matrix"]:
            cm = r["confusion_matrix"]
            print(
                f"  2x2: faithful[correct={cm['faithful_correct']}, incorrect={cm['faithful_incorrect']}] "
                f"unfaithful[correct={cm['unfaithful_correct']}, incorrect={cm['unfaithful_incorrect']}] "
                f"→ correct-but-unfaithful={cm['correct_but_unfaithful_rate']:.1%}"
            )

    results_dir = resolve_path(cfg, cfg["evaluation"]["results_path"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "eval_run.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
