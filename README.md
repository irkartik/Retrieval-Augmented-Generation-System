# RAG Mid-Sem Workroom

Working codebase for *Design and Comparative Evaluation of Retrieval Strategies in a RAG System for Document Question-Answering* (BITS 2024TM93683).

**Start here, in order:**
1. This file — what each piece is.
2. `RUNBOOK.md` — step-by-step commands to actually run it.
3. `Midsem_Plan.md` — the day-by-day tracker for this deliverable.

Full project context lives one level up: `../../../Dissertation_End_to_End_Plan.md` (master plan) and `../../../Examiner_Feedback_Response.md` (why the closed-book baseline and multi-topic sampling exist).

## Directory map

```
workroom/
├── README.md, RUNBOOK.md, Midsem_Plan.md   ← docs (you are here)
├── requirements.txt, .env.example, .python-version, .gitignore
│
├── config/
│   └── default.yaml       ← the ONE file that drives every run: topics, chunk size,
│                             top_k, embedding model, LLM provider. Change behavior here,
│                             not in code.
│
├── rag/                    ← library code (no CLI, just building blocks, imported by scripts/ and demo/)
│   ├── ingestion.py         load SQuAD JSON, filter to the 3 locked topics
│   ├── chunking.py           split paragraphs into token-sized chunks
│   ├── embeddings.py          local sentence-transformers embedder
│   ├── vectorstore.py          build/load the Chroma index
│   ├── retrievers/              one class per strategy behind a common interface
│   │   ├── dense.py               ✅ implemented — baseline semantic search
│   │   ├── closed_book.py          ✅ implemented — no-retrieval baseline (examiner-feedback fix)
│   │   └── (bm25/hybrid/rerank land here for the Final comparative study — not yet built)
│   ├── generation.py            calls the LLM with retrieved chunks, returns answer + citations
│   └── llm_providers.py          swaps openai <-> anthropic based on config only
│
├── scripts/                ← runnable commands, each does ONE step of the pipeline
│   ├── sample_ground_truth.py   step 1: freeze the 450-question test set → data/ground_truth.json
│   ├── build_index.py            step 2: ingest → chunk → embed → data/chroma_db/
│   ├── run_query.py               step 3: ask one question, get a cited answer (needs an LLM key)
│   └── run_eval.py                  step 4: harness — EM/F1 + latency, dense vs. closed-book (needs an LLM key)
│
├── demo/
│   └── streamlit_app.py    ← the interactive chat demo (viva-ready UI), needs an LLM key
│
├── data/                    ← generated artifacts (not code)
│   ├── ground_truth.json     the frozen 450-question set — COMMIT this once generated
│   └── chroma_db/              the vector index — gitignored, rebuild anytime with build_index.py
│
├── results/                 ← eval_run.json outputs land here, for report tables/charts
└── report/                    ← Mid-Sem report draft (md → pdf) goes here
```

**The mental model:** `config/default.yaml` is the dial-board. `rag/` is the engine (never run directly). `scripts/` and `demo/` are the things you actually execute, and they're thin — they just wire `rag/` pieces together in the order the pipeline needs. `data/`, `results/`, `report/` are where output lands, not where you write code.

## Locked decisions (see `Midsem_Plan.md` §5)

| Decision | Value |
| :--- | :--- |
| Orchestration | LangChain |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector store | ChromaDB (local, persisted to `data/chroma_db/`) |
| LLM (generation + judge) | OpenAI `gpt-4o-mini` (config-swappable to Anthropic — see `RUNBOOK.md` step 5) |
| Topics (SQuAD dev split) | `Nikola_Tesla`, `Amazon_rainforest`, `Super_Bowl_50` — distinct domains (science/biography, environment, sports) |
| Ground-truth sample | 450 questions, ~150/topic, fixed seed (`config/default.yaml: seed`) |
| Chunk size / top-k default | 256 tokens / top-k 5 (256↔512 and top-k 3↔5 swept later for the Final study) |

## What's implemented vs. stubbed (honest status as of 7 Sep 2026)

**Working, verified locally (no API key required):**
- SQuAD ingestion + topic filtering, token-based chunking, local embeddings + Chroma indexing
- Ground-truth freezing script (`data/ground_truth.json`, 450 Qs / 150 per topic, frozen)
- Harness scoring internals (EM/F1, citation-stripping, cost estimate, 2×2 matrix) — unit-checked

**Implemented, needs an LLM API key to run end-to-end (see `RUNBOOK.md` step 5):**
- Dense retriever + closed-book baseline, generation with citations (`run_query.py`, `streamlit_app.py`)
- Full harness (`run_eval.py`): EM/F1 correctness + **RAGAS faithfulness** + latency + **approx cost** + the **2×2 faithfulness×correctness matrix** (incl. correct-but-unfaithful rate) + per-topic breakdown

**Stubbed — raises `NotImplementedError` (Final-report work):**
- BM25 / hybrid / re-ranking retrievers
- Paired statistics (McNemar / bootstrap / Wilcoxon) + Cohen's κ judge validation

For exact commands, go to `RUNBOOK.md`.
