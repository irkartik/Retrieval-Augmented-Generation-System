"""Loads SQuAD JSON and slices it down to the locked topic set."""
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paragraph:
    doc_id: str
    title: str
    context: str


@dataclass
class QAPair:
    qid: str
    title: str
    question: str
    answers: list[str]
    context: str


def load_squad(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def select_topics(squad_data: dict, topics: list[str]) -> list[dict]:
    wanted = set(topics)
    selected = [entry for entry in squad_data["data"] if entry["title"] in wanted]
    missing = wanted - {entry["title"] for entry in selected}
    if missing:
        raise ValueError(f"Topics not found in corpus file: {sorted(missing)}")
    return selected


def iter_paragraphs(selected_topics: list[dict]) -> list[Paragraph]:
    paragraphs = []
    for entry in selected_topics:
        for i, para in enumerate(entry["paragraphs"]):
            paragraphs.append(
                Paragraph(doc_id=f"{entry['title']}::{i}", title=entry["title"], context=para["context"])
            )
    return paragraphs


def iter_qas(selected_topics: list[dict]) -> list[QAPair]:
    qas = []
    for entry in selected_topics:
        for para in entry["paragraphs"]:
            for qa in para["qas"]:
                answers = list(dict.fromkeys(a["text"] for a in qa["answers"]))  # dedup, keep order
                qas.append(
                    QAPair(
                        qid=qa["id"],
                        title=entry["title"],
                        question=qa["question"],
                        answers=answers,
                        context=para["context"],
                    )
                )
    return qas
