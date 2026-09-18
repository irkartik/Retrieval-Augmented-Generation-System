# Retrieval-Augmented-Generation-System

Codebase for *Design and Comparative Evaluation of Retrieval Strategies in a RAG System for Document Question-Answering* (BITS 2024TM93683).

## Directory map

```
.
├── README.md
├── requirements.txt, .env.example, .python-version, .gitignore
│
├── config/
│   └── default.yaml       ← the ONE file that drives every run: corpus path, topics, chunk size,
│                             top_k, embedding model, LLM provider. Change behavior here, not in code.
│
├── rag/                    ← library code (no CLI, just building blocks, imported by scripts/ and demo/)
│   ├── ingestion.py         load SQuAD JSON, filter to the locked topics
│   ├── chunking.py           split paragraphs into token-sized chunks
│   ├── embeddings.py          local sentence-transformers embedder
│   ├── vectorstore.py          build/load the Chroma index
│   ├── retrievers/              one class per strategy behind a common interface
│   │   ├── dense.py               ✅ implemented — baseline semantic search
│   │   ├── closed_book.py          ✅ implemented — no-retrieval baseline
│   │   └── (bm25/hybrid/rerank land here for the comparative study — not yet built)
│   ├── generation.py            calls the LLM with retrieved chunks, returns answer + citations
│   ├── faithfulness.py           RAGAS faithfulness scoring
│   ├── pricing.py                 approx cost estimate per run
│   └── llm_providers.py          swaps openai <-> anthropic based on config only
│
├── scripts/                ← runnable commands, each does ONE step of the pipeline
│   ├── sample_ground_truth.py   step 1: freeze the test question set → data/ground_truth.json
│   ├── build_index.py            step 2: ingest → chunk → embed → data/chroma_db/
│   ├── run_query.py               step 3: ask one question, get a cited answer (needs an LLM key)
│   ├── run_eval.py                  step 4: harness — EM/F1 + latency, dense vs. closed-book (needs an LLM key)
│   └── add_faithfulness.py           step 5: backfill RAGAS faithfulness scores onto an eval run
│
├── demo/
│   └── streamlit_app.py    ← interactive chat demo, needs an LLM key
│
├── data/
│   ├── squad/                the raw corpus — dev-v1.1.json (used) + train-v1.1.json
│   ├── ground_truth.json     the frozen question/answer set — commit this once generated
│   └── chroma_db/              the vector index — gitignored, rebuild anytime with build_index.py
│
└── results/                 ← eval_run.json outputs land here, for report tables/charts
```

**The mental model:** `config/default.yaml` is the dial-board. `rag/` is the engine (never run directly). `scripts/` and `demo/` are the things you actually execute, and they're thin — they just wire `rag/` pieces together in the order the pipeline needs. `data/` and `results/` are where output lands, not where you write code.

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY / ANTHROPIC_API_KEY
```

## Current configuration (see `config/default.yaml` for the source of truth)

| Setting | Value |
| :--- | :--- |
| Orchestration | LangChain |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector store | ChromaDB (local, persisted to `data/chroma_db/`) |
| LLM (generation + judge) | config-swappable between OpenAI and Anthropic |
| Corpus | SQuAD dev split (`data/squad/dev-v1.1.json`) |
| Topics | `Nikola_Tesla`, `Amazon_rainforest`, `Black_Death`, `French_and_Indian_War` |
| Ground-truth sample | 450 questions, stratified per topic, fixed seed |
| Chunk size / top-k default | 256 tokens / top-k 5 |

## What's implemented vs. stubbed

**Working, verified locally (no API key required):**
- SQuAD ingestion + topic filtering, token-based chunking, local embeddings + Chroma indexing
- Ground-truth freezing script (`data/ground_truth.json`)
- Harness scoring internals (EM/F1, citation-stripping, cost estimate, 2×2 matrix)

**Implemented, needs an LLM API key to run end-to-end:**
- Dense retriever + closed-book baseline, generation with citations (`run_query.py`, `streamlit_app.py`)
- Full harness (`run_eval.py`): EM/F1 correctness + RAGAS faithfulness + latency + approx cost + the 2×2 faithfulness×correctness matrix (incl. correct-but-unfaithful rate) + per-topic breakdown

**Stubbed — raises `NotImplementedError`:**
- BM25 / hybrid / re-ranking retrievers
- Paired statistics (McNemar / bootstrap / Wilcoxon) + Cohen's κ judge validation

## Running the pipeline

```
python scripts/sample_ground_truth.py   # freeze the test set
python scripts/build_index.py           # ingest, chunk, embed, index
python scripts/run_query.py "<question>"
python scripts/run_eval.py              # full evaluation harness
```
