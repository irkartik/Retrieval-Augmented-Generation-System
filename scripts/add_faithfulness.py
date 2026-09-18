"""Post-process eval results to add faithfulness scores to all retrieval-based strategies.

Reads eval_run.json (without faithfulness), scores all strategies with retrieved_chunks
for faithfulness using stored question and chunk data, and writes the updated JSON back.

Skips strategies without retrieved chunks (e.g., closed_book) automatically.

Usage: python -m scripts.add_faithfulness
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from rag.config import load_config, resolve_path
from rag.faithfulness import build_faithfulness_metric, score_faithfulness
from rag.retrievers.base import RetrievedChunk


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def confusion_matrix(per_question: list[dict]) -> dict | None:
    """2x2 faithfulness x correctness matrix over questions that were faithfulness-scored."""
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

def score_strategy(strategy_name: str, strategy_results: dict, faith_metric, faith_threshold: float) -> None:
    """Score all answers in a strategy for faithfulness."""
    per_question = strategy_results["per_question"]

    print(f"[{strategy_name}] Scoring {len(per_question)} answers...", flush=True)

    for i, qa_result in enumerate(per_question):
        if (i + 1) % 50 == 0:
            print(f"  [{strategy_name}] Q {i+1}/{len(per_question)}", flush=True)

        question = qa_result.get("question", "")
        answer = qa_result.get("prediction", "")

        # Reconstruct chunks from stored data
        chunk_dicts = qa_result.get("retrieved_chunks", [])
        chunks = [RetrievedChunk(text=c["text"], title=c["title"], doc_id=c["doc_id"], score=c["score"])
                  for c in chunk_dicts]

        # Try to score faithfulness
        try:
            faith_score = score_faithfulness(faith_metric, question, answer, chunks)
            faithful = faith_score >= faith_threshold
        except Exception as e:
            print(f"  Warning [{strategy_name}] Q {i+1}: {e}")
            faith_score = None
            faithful = None

        qa_result["faithfulness"] = faith_score
        qa_result["faithful"] = faithful

    # Recalculate aggregates
    faith_scores = [r["faithfulness"] for r in per_question if r["faithfulness"] is not None]
    strategy_results["mean_faithfulness"] = _mean(faith_scores) if faith_scores else None
    strategy_results["confusion_matrix"] = confusion_matrix(per_question)

    # Print summary
    print(f"\n[{strategy_name}] Complete:")
    print(f"  Mean faithfulness: {strategy_results['mean_faithfulness']}")
    if strategy_results["confusion_matrix"]:
        cm = strategy_results["confusion_matrix"]
        print(f"  2x2 matrix: faithful[correct={cm['faithful_correct']}, incorrect={cm['faithful_incorrect']}] "
              f"unfaithful[correct={cm['unfaithful_correct']}, incorrect={cm['unfaithful_incorrect']}]")
        print(f"  Correct-but-unfaithful rate: {cm['correct_but_unfaithful_rate']:.1%}")


def main():
    print("Starting faithfulness scoring...", flush=True)

    load_dotenv()
    print("Loaded environment config", flush=True)

    cfg = load_config()
    print("Loaded evaluation config", flush=True)

    # Load existing results
    results_path = resolve_path(cfg, cfg["evaluation"]["results_path"])
    eval_file = results_path / "eval_run.json"

    print(f"Loading eval results from {eval_file}...", flush=True)
    with open(eval_file) as f:
        results = json.load(f)
    print(f"Loaded results for {len(results)} strategies", flush=True)

    # Find all strategies with retrieved chunks (skip closed_book and others without retrieval)
    strategies_to_score = []
    for strategy_name, strategy_results in results.items():
        per_question = strategy_results.get("per_question", [])
        if per_question and "retrieved_chunks" in per_question[0]:
            strategies_to_score.append(strategy_name)

    if not strategies_to_score:
        print("No strategies with retrieved_chunks found. Skipping.")
        return

    print(f"Scoring faithfulness for strategies: {', '.join(strategies_to_score)}\n", flush=True)

    # Build faithfulness metric (once, reuse for all strategies)
    print("Building faithfulness metric (this may take a moment)...", flush=True)
    faith_metric = build_faithfulness_metric(cfg)
    print("Faithfulness metric ready\n", flush=True)

    faith_threshold = cfg["evaluation"].get("faithfulness_threshold", 0.5)

    # Score each strategy
    for strategy_name in strategies_to_score:
        score_strategy(strategy_name, results[strategy_name], faith_metric, faith_threshold)

    # Save updated results
    print("\nSaving updated results...", flush=True)
    with open(eval_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Wrote updated results to {eval_file}", flush=True)


if __name__ == "__main__":
    main()
