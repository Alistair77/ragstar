# Hybrid Search RAG

A Retrieval-Augmented Generation (RAG) system over a folder of internal documents, combining **vector search** (Pinecone) with **keyword search** (BM25), merged via **Reciprocal Rank Fusion**, refined with **Cohere reranking**, and answered by a **local LLM** (Ollama) with inline citations.

Built as a learning project — this README doubles as a beginner-friendly explanation of every concept involved.

---

## The Big Picture: What Problem Does RAG Solve?

An LLM only knows what it learned during training. It has **never seen your internal documents** — your onboarding guide, incident-response policy, internal wiki. Two bad options if you want it to answer questions about those docs:

- **Fine-tune the model** — slow, expensive, and it may still hallucinate instead of citing real text.
- **Paste every document into every prompt** — breaks down with thousands of documents (too much text, too slow, too expensive).

**RAG is the practical middle ground:**

> Before asking the LLM anything, first *search* your documents for the few paragraphs most relevant to the question, then hand only those paragraphs to the LLM and say "answer using this."

A RAG system is really two systems glued together:
1. A **search engine** over your documents (most of the code here).
2. An **LLM call** at the end that reads the search results and writes an answer.

### Why "Hybrid" Search?

This project searches two different ways at once and combines the results:

| Search type | Good at | Bad at |
|---|---|---|
| **Vector/semantic search** (Pinecone) | Understanding *meaning* — "car" matches "automobile" with zero shared words | Exact terms — acronyms, product codes, IDs can get lost |
| **Keyword search** (BM25) | Exact word matches — IDs, names, jargon | Meaning — won't connect "car" and "automobile" |

Combining them catches cases either one alone would miss. Example: a user asks about "SEV-1" but the doc says "Severity-1" — BM25 misses it (no shared tokens), vector search catches it (same meaning).

---

## Architecture

There are **two separate flows**:

### Flow A — Ingestion (runs once, whenever docs change)

```
sample_docs/*.txt
      │
      ▼
  chunk_documents()        ← RecursiveCharacterTextSplitter, ~500 chars, 50 overlap
      │
      ▼
  LocalEmbeddings           ← sentence-transformers, text → 384-dim vector
      │
      ▼
  Pinecone.upsert()         ← vectors + text + source metadata stored in the cloud
```

### Flow B — Query (runs on every API call)

```
POST /query {"question": "..."}
      │
      ▼
┌─────────────────────┬─────────────────────┐
│  Vector search       │   BM25 search        │   ← run independently, same query
│  (Pinecone, top 20)  │   (in-memory, top 20)│
└──────────┬───────────┴───────────┬──────────┘
           └──────────┬────────────┘
                       ▼
          Reciprocal Rank Fusion (RRF)      ← merges 2 ranked lists into 1
                       │
                       ▼
          Cohere rerank (cross-encoder)     ← re-scores the 20, keeps top 5
                       │
                       ▼
          Build prompt: [SOURCE 1]...[SOURCE 5] + question
                       │
                       ▼
          Ollama (local LLM) generates answer with [Source N] citations
                       │
                       ▼
          JSON response: {question, answer, sources[]}
```

Mental model: **Flow A is "build the library." Flow B is "answer a question using the library."**

---

## Stage-by-Stage Explanation

### Stage 1 — Ingestion (`stage1_ingestion.py`, `embeddings.py`)

**Chunking.** You can't embed a whole 10-page document as one unit — embedding models have input limits, and one giant vector representing an entire document is too "blurry." If someone asks about paragraph 7, you want to retrieve *just paragraph 7*. So each doc is split into small overlapping **chunks**:

- **"Recursive"** splitting cuts at natural boundaries first: paragraph breaks → line breaks → sentence ends → spaces — avoiding severing a sentence mid-thought.
- **Overlap (50 chars)** exists because chunk boundaries are arbitrary. If a key sentence falls right across a cut point ("...the incident must be escalated" | "within 15 minutes"), neither chunk alone makes sense. Repeating the tail of each chunk at the start of the next keeps critical sentences intact somewhere.

Each chunk gets a stable MD5-hash ID (filename + position + first 50 chars), so re-ingesting the same docs *updates* Pinecone entries instead of duplicating them.

**Embedding.** An **embedding** is a list of numbers (here 384) representing the *meaning* of text. Texts with similar meaning become nearby points in 384-dimensional space. This project uses `all-MiniLM-L6-v2` from sentence-transformers — small, free, runs locally on CPU, no API key.

**Pinecone** is a vector database. Its one job: given a query vector, instantly find the K stored vectors closest to it, even across millions of entries. It stores each vector alongside the original chunk text and source metadata.

### Stage 2 — BM25 Keyword Index (`core.py`)

**BM25** is a decades-old, battle-tested relevance-scoring algorithm — no neural network, just statistics. Its score combines three intuitive ideas:

1. **Term frequency** — chunks mentioning your query words more often score higher.
2. **Inverse document frequency** — a shared word that's *rare* across the corpus (like "escalation") counts far more than a common one (like "the") that appears everywhere.
3. **Length normalization** — long chunks aren't unfairly favored just for having more chances to match.

Chunks sharing zero words with the query score 0 and are discarded.

Key contrast: Pinecone finds things that *mean* the same; BM25 finds things that *say* the same words. They're fully independent searches over the same chunks — which is what makes combining them valuable.

### Stage 3 — Hybrid Retrieval + Reciprocal Rank Fusion (`rrf.py`)

Two ranked lists for the same query — but **their scores aren't comparable**. Pinecone gives cosine similarity (0–1); BM25 gives unbounded statistics (could be 3.7 or 47.2). You can't just add them.

**RRF's trick: throw away the raw scores and use only each item's rank position.**

```
rrf_score = 1 / (k + rank)      # k = 60, a standard smoothing constant
```

Rank #1 contributes 1/61, rank #2 contributes 1/62, and so on — a smooth decaying curve regardless of the underlying score scale. If a chunk appears in **both** lists, its two contributions are **added** — "both independent methods think this is relevant" is a strong signal that pushes it to the top of the merged list.

### Stage 4 — Cohere Reranking (`core.py`)

Why sort *again* after RRF? Because of a fundamental accuracy/speed trade-off:

- **Bi-encoder** (Pinecone): query and documents embedded *separately*, compared with simple math. Fast enough for millions of chunks because document vectors are precomputed.
- **Cross-encoder** (Cohere rerank): query and candidate fed into the neural network *together*, letting it directly compare specific words and phrases. Far more accurate — but requires a full model pass per document per query, so it's too slow to run over an entire corpus.

The production pattern: **retrieve loosely and cheaply over everything, then rerank tightly and expensively over just the top candidates.** Pinecone + BM25 narrow everything down to ~20 plausible chunks; Cohere carefully re-scores those 20 and keeps the best 5.

### Stage 5 — Generation with Citations (`core.py`)

The top-5 chunks are labeled `[SOURCE 1]`...`[SOURCE 5]` and pasted into the prompt (**context injection**). The prompt instruction does three deliberate jobs:

1. **"Answer using ONLY the provided sources"** — called **grounding**. Left alone, an LLM will generate plausible-sounding but fabricated answers (**hallucination**). Restricting it to the provided text dramatically reduces that risk.
2. **"Cite the source as [Source N]"** — creates **traceability**. Readers can verify which chunk backs each claim instead of trusting the model blindly.
3. **"If the sources lack the information, say what's missing"** — explicitly permits "I don't know." This one sentence prevents a huge share of confident-but-wrong answers.

Generation runs on a **local Ollama model** — free, private, no API key. Swapping LLM providers (Claude ↔ Ollama ↔ OpenAI) is usually a tiny change because they all converged on the same "list of `{role, content}` messages in → text out" interface.

### Config & API Layer (`config.py`, `main.py`)

- `Settings` (pydantic-settings) reads `.env`, validates types, and provides one typed settings object to every module. Required keys with no default (**failing fast**): if `.env` is missing them, the app crashes immediately with a clear error — far better than failing confusingly mid-request later.
- FastAPI exposes the pipeline as `POST /query`. Pydantic models validate incoming/outgoing JSON automatically.
- The **BM25 index is rebuilt at server startup** because it lives only in RAM and vanishes when the process stops — unlike Pinecone, which persists in the cloud regardless of your server's state.
- Adding new documents requires **re-running ingestion** (Flow A). Query time never re-reads raw files — new docs are invisible until ingested.

---

## Tech Stack (Free Tier / Local)

| Stage | Tool | Cost |
|---|---|---|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, local) | Free — runs on CPU |
| Vector store | Pinecone Starter tier | Free tier |
| Keyword index | `rank_bm25` (local) | Free |
| Reranking | Cohere trial key | Free trial, rate-limited |
| Generation | Ollama (local) | Free |

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install Ollama (https://ollama.com) and pull a model, e.g.:
ollama pull qwen2.5:3b
```

Create a `.env` file (see `.env.example`):

```
PINECONE_API_KEY=pcsk_xxxxx     # free tier: pinecone.io
PINECONE_INDEX_NAME=hybrid-rag
COHERE_API_KEY=xxxxx            # free trial: dashboard.cohere.com
```

## Usage

```bash
# One-shot: ingest docs + run the 3 sample questions end-to-end
python run_all.py

# Or step by step:
python stage1_ingestion.py     # Flow A: chunk, embed, upsert to Pinecone
python test_rag.py             # full pipeline test with verbose stage output

# Run the API server
python main.py                 # then POST {"question": "..."} to localhost:8000/query

# Unit tests (no API keys needed)
python test_rrf.py
```

## Project Structure

```
hybrid-rag/
├── config.py             # typed settings from .env (pydantic-settings)
├── embeddings.py         # local sentence-transformers embeddings
├── stage1_ingestion.py   # Flow A: load → chunk → embed → Pinecone
├── rrf.py                # standalone Reciprocal Rank Fusion (unit-tested)
├── core.py               # Stages 2–5: BM25, hybrid search, rerank, generation
├── main.py               # FastAPI /query endpoint
├── test_rrf.py           # RRF unit tests (offline)
├── test_rag.py           # full-pipeline test, 3 sample questions
├── run_all.py            # one-shot runner
└── sample_docs/          # toy document set
```

## Known Issues & Improvement Ideas

Roughly ordered by learning value per unit effort:

1. **RRF rank-offset bug** 🔴 — in `rrf.py`, BM25 results are added with `rank_offset=len(vector_results)`, so BM25's best hit is treated as rank ~21 instead of rank 1. This systematically penalizes every BM25 result and biases the merge toward vector search. Fix: use `rank_offset=0` for both lists.
2. **Evaluation** 🟠 — the most valuable RAG skill to learn next. Build a test set of questions with known correct source chunks, measure retrieval hit-rate@5 and answer faithfulness (LLM-as-judge). Turns "it feels better" into "hit-rate went from 60% to 85%."
3. **Robustness** 🟡 — no retry/backoff on external calls (a network hiccup crashes the request); orphaned vectors linger in Pinecone when docs shrink; ingestion and BM25-index-build independently re-read the raw files and can drift.
4. **Features** 🟢 — streaming answers, programmatic citation verification, semantic chunking (compare against fixed-size using your eval set), PDF/DOCX loaders.
