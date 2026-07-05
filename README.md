# Hybrid Search RAG

A Retrieval-Augmented Generation (RAG) system that searches documents using **both** vector search (meaning) and keyword search (exact words), merges the results, and generates answers with citations using a local LLM.

**Zero API keys required** — everything runs locally on your machine.

---

## Quick Start

```bash
# One-time setup (already done in this project)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Make sure Ollama has the model
ollama pull qwen3b-128k

# Run the demo — 5 questions with full stage-by-stage output
python local_rag.py

# Run the eval suite — 10 questions, retrieval metrics, faithfulness check
python local_rag.py --eval

# Launch the web UI — interactive, visual, shows every retrieval stage
python demo_app.py
# → open http://127.0.0.1:8100
```

---

## Architecture (5-Stage Pipeline)

```
User Question
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Stage 2:  Vector Search (semantic meaning)          │
│            all-MiniLM-L6-v2 → cosine similarity      │
│                                                      │
│  Stage 2:  BM25 Search (exact keyword match)         │
│            term frequency × IDF stats                │
└────────────────────┬──────────────────┬──────────────┘
                     └──────┬───────────┘
                            ▼
┌──────────────────────────────────────────────────────┐
│  Stage 3:  Reciprocal Rank Fusion (RRF)              │
│            merges 2 ranked lists by position          │
│            score = 1/(60 + rank) from each list      │
└────────────────────────┬─────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────┐
│  Stage 4:  Cross-Encoder Rerank                      │
│            ms-marco-MiniLM-L-6-v2 (local)            │
│            re-scores top 10, keeps best 4            │
└────────────────────────┬─────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────┐
│  Stage 5:  LLM Generation with citations             │
│            qwen3b-128k via local Ollama              │
│            builds prompt with [Source N] context     │
│            answer + faithfulness_verification        │
└──────────────────────────────────────────────────────┘
```

### Stage 1 — Ingestion (runs once on startup)

Documents → chunked at 500 characters (50 overlap) → each chunk embedded as a 384-dim vector → stored in both a NumPy matrix (cosine similarity) and a BM25 index.

### Stage 2 — Dual Retrieval (runs on every query)

| Retriever | What it catches | Blind spot |
|---|---|---|
| **Vector search** (semantic) | Synonyms, paraphrases, "stipend" ↔ "reimbursement" | Exact terms, IDs, codes |
| **BM25** (keyword) | Exact words, "SEV-1", "$1,500", "90 days" | Synonyms, meaning |

### Stage 3 — RRF Fusion

Throws away the raw scores (cosine similarity vs BM25 statistics — not comparable) and uses only **rank position**. Each item scores `1/(60 + rank)`. If one chunk appears in **both** lists, its scores add up — "both independent methods think this matters" pushes it to the top.

### Stage 4 — Cross-Encoder Rerank

Cross-encoders are more accurate than vector search but too slow to run over millions of chunks. The production pattern: **retrieve loosely over everything, rerank tightly over the top candidates.**

### Stage 5 — Generation + Faithfulness Check

The prompt does three things:
1. **"Answer using ONLY the provided sources"** — prevents hallucination
2. **"Cite as [Source N]"** — creates traceability
3. **"If sources are missing, say so"** — permits "I don't know"

After generation, a **faithfulness verification** step (LLM-as-Judge) checks each cited claim against its source document. Unsupported claims are flagged.

---

## How the Two Retrievers Compete and Cooperate

The key insight: **you want two independent sources of signal, not one.**

When a user asks about "SEV-1":

- **BM25 wins** — the term "SEV-1" is rare in the corpus, so it scores very high on IDF. Vector search finds it too (it knows "SEV-1" means "critical incident"), but BM25 is more confident about exact matches.

When a user asks "How much is the home office stipend?":

- **Vector search wins** — "stipend" isn't in every chunk, but the embedding connects "stipend", "allowance", "reimbursement" as related concepts. BM25 misses if the exact word "stipend" isn't in the chunk.

When a user asks "Can I claim both internet and co-working?":

- **Both win** — "internet" is an exact keyword match for BM25, and the semantic embedding connects the exclusion clause ("cannot claim both") to the question about stacking benefits. The chunk appears in both lists, gets double RRF credit, and rockets to #1.

---

## Demo Data

The `demo_docs/` folder contains 4 Markdown documents about a fictional company called **Nimbus Robotics**:

| File | Topics |
|---|---|
| `employee_handbook.md` | Remote work, home office stipend ($1,500), internet reimbursement ($75/mo), co-working ($300/mo), PTO, parental leave, learning budget ($2,000/yr) |
| `engineering_onboarding.md` | First-week setup, code review culture (respond within 1 business day), deployment, PR guidelines |
| `security_incident_policy.md` | SEV-1/2/3/4 definitions, escalation process, postmortems, GDPR data breach notification (72 hours) |
| `travel_and_expense_policy.md` | Air travel (economy/premium/business), hotels ($250/$350), meals ($80/day), non-reimbursable expenses |

23 chunks total across 4 documents.

---

## Demo Queries to Try

These 5 sample questions are built in:

1. **"How much is the home office stipend and when can I use it?"**
2. **"What do I do when a SEV-1 incident happens?"**
3. **"Can I expense a business class flight to Tokyo?"**
4. **"How quickly must reviewers respond to a pull request?"**
5. **"Can I claim both internet reimbursement and a co-working membership?"**

More to test the edges:

6. **"What is the Nimbus learning budget?"** — vector-friendly ($2,000)
7. **"How many days of PTO do I get?"** — cross-doc reasoning (20 vacation + 10 holidays)
8. **"What happens if customer data is leaked?"** — harder, requires escalation + GDPR section
9. **"How much can I expense for a hotel in London?"** — needs "high-cost cities" rule ($350)
10. **"Can I expense wine at dinner?"** — needs "alcohol only during team events" exception

---

## Evaluation Results

Run `python local_rag.py --eval` to reproduce.

### Retrieval Metrics (top-5)

```
Hit-rate@5:  100.0%  (10/10)
MRR@5:        1.000

Easy:    4/4 (100%)
Medium:  3/3 (100%)
Hard:    3/3 (100%)
```

All 10 test questions find their correct source chunk in the top-5 results. The hybrid retrieval (vector + BM25 + RRF) handles easy factual lookups, cross-section reasoning, and hard policy edge cases equally well.

### Faithfulness (LLM-as-Judge)

```
Faithfulness rate:  100% (5/5)
Average score:       1.00
```

Every generated answer's cited claims are fully supported by the source documents. The faithfulness check runs an independent LLM call that extracts each [Source N] citation, finds the corresponding claim in the answer, and verifies it against the source text.

---

## Project Structure

```
hybrid-rag/
├── local_rag.py             # Fully-local pipeline (no API keys)
├── demo_app.py              # Web UI (FastAPI + HTML)
├── eval_rag.py              # Evaluation framework
├── faithfulness.py          # LLM-as-Judge claim verification
├── rrf.py                   # Reciprocal Rank Fusion (standalone)
├── embeddings.py            # sentence-transformers wrapper
├── stage1_ingestion.py      # Cloud ingestion (Pinecone)
├── core.py                  # Cloud pipeline (Pinecone + Cohere)
├── config.py                # Settings from .env
├── main.py                  # FastAPI /query endpoint (cloud)
├── run_all.py               # One-shot: ingest + query
├── sample_docs/             # 3 TXT files (cloud demo)
├── demo_docs/               # 4 MD files (local demo)
└── requirements.txt
```

---

## How to Run Each Mode

| Command | What it does | API keys needed? |
|---|---|---|
| `python local_rag.py` | 5 demo questions, stage-by-stage, with faithfulness check | No |
| `python local_rag.py --eval` | 10-question eval with retrieval + faithfulness metrics | No |
| `python local_rag.py --no-verify` | Same as default but skip faithfulness check | No |
| `python demo_app.py` | Web UI at http://127.0.0.1:8100 | No |
| `python run_all.py` | Ingest + quick query loop | Yes (Pinecone + Cohere) |
| `python main.py` | FastAPI server at http://0.0.0.0:8000 | Yes (Pinecone + Cohere) |

---

## How Improvements Changed the System

The original project was a working RAG system. Two improvements were added:

### 1. Evaluation Framework (`eval_rag.py`)

Before: no systematic way to measure retrieval quality. "It feels better" was the only metric.

After: a golden dataset of 10 questions with expected source phrases, difficulty labels, and two metrics:
- **Hit-rate@N** — was the right chunk retrieved? (100%)
- **MRR@N** — where did it rank? (1.000)
- **Faithfulness rate** — are the answers grounded? (100%)

The eval dataset revealed and fixed a whitespace normalization bug: chunking inserts newlines that break substring matching in eval checks. The fix (`_normalize()`) collapses whitespace before comparing.

### 2. Faithfulness Verification (`faithfulness.py`)

Before: the pipeline generated answers with citations but no verification. The first eval run showed that the LLM sometimes hallucinated (answered "Yes, you can expense business class" when the source says "premium economy only").

After: every answer is checked by an independent LLM-as-Judge call that:
- Parses all `[Source N]` citations
- Extracts the associated claims from the answer
- Verifies each claim against the source text
- Returns a score (0.0–1.0) and a list of unsupported claims
- Flunks the answer if any claim is unsupported

This catches hallucinations at generation time instead of shipping them to the user.

---

## Tech Stack

| Stage | Tool | Runs on |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | CPU |
| Vector index | In-memory NumPy matrix | RAM |
| Keyword index | `rank_bm25` (BM25Okapi) | RAM |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CPU |
| LLM | Ollama (`qwen3b-128k`) | CPU/GPU |
| Web UI | FastAPI + vanilla HTML/JS | localhost |
| Eval | Custom framework (LLM-as-Judge) | CPU |
| Faithfulness | LLM-as-Judge via Ollama | CPU |
