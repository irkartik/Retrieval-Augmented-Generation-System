"""Freezes the 300-500-question, 2-3-topic ground-truth set (run once; commit the output).

Usage: python -m scripts.sample_ground_truth
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import load_config, resolve_path
from rag.ingestion import iter_qas, load_squad, select_topics


def main():
    cfg = load_config()
    corpus_cfg = cfg["corpus"]

    squad_path = resolve_path(cfg, corpus_cfg["path"])
    squad_data = load_squad(squad_path)
    selected = select_topics(squad_data, corpus_cfg["topics"])
    qas = iter_qas(selected)

    rng = random.Random(cfg["seed"])
    sample_size = corpus_cfg["sample_size"]
    n_topics = len(corpus_cfg["topics"])
    per_topic_target = sample_size // n_topics

    sampled = []
    for topic in corpus_cfg["topics"]:
        topic_qas = [qa for qa in qas if qa.title == topic]
        rng.shuffle(topic_qas)
        take = min(per_topic_target, len(topic_qas))
        sampled.extend(topic_qas[:take])

    rng.shuffle(sampled)

    out_path = resolve_path(cfg, corpus_cfg["ground_truth_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            [
                {
                    "id": qa.qid,
                    "topic": qa.title,
                    "question": qa.question,
                    "gold_answers": qa.answers,
                    "gold_context": qa.context,
                }
                for qa in sampled
            ],
            f,
            indent=2,
        )

    counts = {t: sum(1 for qa in sampled if qa.title == t) for t in corpus_cfg["topics"]}
    print(f"Wrote {len(sampled)} questions to {out_path}")
    print(f"Per-topic breakdown: {counts}")


if __name__ == "__main__":
    main()
