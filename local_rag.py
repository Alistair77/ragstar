"""
FULLY-LOCAL Hybrid RAG — no API keys, no signups, no internet (after models cache).

Same 5-stage pipeline as core.py, but the two cloud pieces are swapped for local ones
so you can actually SEE it run:

    core.py (cloud)                  local_rag.py (this file)
    ----------------------------     ----------------------------------------
    Pinecone vector DB          -->  in-memory NumPy cosine similarity
    Cohere rerank (cross-enc.)  -->  local sentence-transformers CrossEncoder
    OpenAI embeddings           -->  local sentence-transformers (already local)
    Claude API                  -->  local Ollama (already local)

Everything else — chunking, BM25, Reciprocal Rank Fusion, the citation prompt —
is identical in spirit to the real project. Run it:  python local_rag.py
"""

import textwrap
from functools import lru_cache
from pathlib import Path

import numpy as np
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rrf import reciprocal_rank_fusion
from faithfulness import verify_faithfulness

# ── Settings (plain constants — no .env needed) ──────────────────────
DOCS_DIR = Path(__file__).parent / "demo_docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_HYBRID = 10          # how many candidates each retriever returns
TOP_K_RERANK = 4           # how many survive reranking and reach the LLM
# Refuse to answer when the best reranked chunk scores below this floor — nothing
# retrieved is relevant enough to ground an answer. Measured on the demo corpus:
# real answers score +6..+9 (a weak-but-valid match hit -1.1); unanswerable
# questions all scored ~ -11. -6.0 sits in the empty gap between them.
# ponytail: fixed floor; make it per-corpus if a new document set shifts the scale.
REFUSE_BELOW_RERANK = -6.0
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Tested qwen2.5:0.5b (fast) — too weak, said "I could not find that" on questions
# whose answer WAS retrieved. qwen3b is the smallest model that answers reliably.
OLLAMA_MODEL = "qwen3b-128k"

# Speed switch. When False we SKIP the cross-encoder entirely — the model is
# never loaded (faster startup) and never runs (faster per-query). BUT: testing
# showed rerank does real work here — it lifts the answer-bearing chunk into the
# top-4 that reach the LLM. With it off, the right chunk falls out of the window
# and the model answers "I don't know". So we keep it ON. Flip to False only if
# you accept lower answer quality for a faster startup.
USE_RERANK = True

# Query rewriting: one small LLM call that cleans the question BEFORE retrieval —
# fixes typos and expands abbreviations ("PTO" -> "paid time off"), so BM25 has
# real words to match and the embedding sees a well-formed question. Costs one
# extra LLM call per query. Every failure mode falls back to the original query,
# so the worst case is "no improvement", never a broken search.
USE_QUERY_REWRITE = True
# How many distinct questions to remember. Ask the same thing twice and we skip
# both the embedding model AND the rewrite LLM call. 256 covers any demo session
# and costs a few KB. Caches are keyed on the query text only — they stay valid
# when documents change, because neither step reads the documents.
CACHE_SIZE = 256
# A rewrite longer than this multiple of the original means the model explained
# itself instead of rewriting — discard it and keep the user's question.
REWRITE_MAX_GROWTH = 3
# ...but short queries are allowed to grow past that multiple, up to this many
# characters. Expanding an abbreviation makes a short query much longer on
# purpose: "PTO polcy" (9 chars) -> "What is the paid time off policy?" (33) is
# a GOOD rewrite that a bare 3x rule would have thrown away. A real rambling
# answer runs to paragraphs, so it still trips this ceiling.
REWRITE_MAX_CHARS = 120


class LocalHybridRAG:
    def __init__(self):
        print("Loading local models (first run downloads them, then they cache)…")
        self._embedder = SentenceTransformer(EMBED_MODEL)
        # Only pay the cost of loading the cross-encoder if we actually rerank.
        self._reranker = CrossEncoder(RERANK_MODEL) if USE_RERANK else None
        self._ollama = ollama.Client()
        self._chunks: list[dict] = []
        self._matrix: np.ndarray | None = None   # normalized chunk embeddings
        self._bm25: BM25Okapi | None = None

    # ── Stage 1: Ingestion ───────────────────────────────────────────
    def ingest(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = []
        for fpath in sorted(DOCS_DIR.glob("*")):
            if fpath.suffix not in {".md", ".txt", ".rst"}:
                continue
            for i, text in enumerate(splitter.split_text(fpath.read_text("utf-8"))):
                chunks.append({
                    "id": f"{fpath.stem}-{i}",
                    "text": text,
                    "source": fpath.name,
                })
        self._chunks = chunks

        # Vector index: embed every chunk, L2-normalize so dot product = cosine.
        embeddings = self._embedder.encode([c["text"] for c in chunks])
        self._matrix = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Keyword index: BM25 over the same chunks.
        self._bm25 = BM25Okapi([c["text"].lower().split() for c in chunks])

        print(f"Stage 1: ingested {len(chunks)} chunks from "
              f"{len(list(DOCS_DIR.glob('*.md')))} documents\n")
        return chunks

    # ── Stage 0: Query rewriting (runs before any retrieval) ────────
    # Cached: re-asking the same question skips an entire LLM round-trip, which
    # is the most expensive thing in this stage (seconds, vs milliseconds for the
    # embedding). Deterministic anyway at temperature=0, so the cached value is
    # exactly what a fresh call would return.
    @lru_cache(maxsize=CACHE_SIZE)
    def rewrite_query(self, query: str) -> str:
        """Clean the question before retrieval. Returns the ORIGINAL on any doubt.

        Retrieval is only as good as the words it is given. A misspelled or
        abbreviated question ("wat is teh PTO polcy") gives BM25 nothing to match
        and pushes the embedding off target. One cheap LLM call fixes spelling and
        expands abbreviations, and both retrievers then see the cleaned version.

        Every failure path returns `query` unchanged, so a bad rewrite can never
        be worse than no rewrite.
        """
        if not USE_QUERY_REWRITE or not query.strip():
            return query

        prompt = (
            "Rewrite the question below so it is easy to search.\n"
            "- fix spelling mistakes\n"
            "- expand abbreviations (PTO -> paid time off, PR -> pull request)\n"
            "- keep the original keywords, add no new facts\n"
            "Reply with the rewritten question ONLY. No explanation, no quotes.\n\n"
            f"Question: {query}\n\n"
            "Rewritten:"
        )
        try:
            resp = self._ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )
            out = resp["message"]["content"]
            # Some models emit a reasoning block first — keep only what follows it.
            if "</think>" in out:
                out = out.split("</think>")[-1]
            # Take the first real line; models like to add commentary underneath.
            out = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
            out = out.strip('"').strip("'").strip()
        except Exception:
            return query          # Ollama down / model missing → search as typed

        # Reject a rewrite that came back empty or that rambled instead of
        # rewriting. Length is a crude but reliable tell for the second case.
        # The absolute floor matters: without it, expanding an abbreviation in a
        # short query ("PTO polcy" -> "What is the paid time off policy?") blows
        # past 3x and the best rewrites get thrown away.
        limit = max(len(query) * REWRITE_MAX_GROWTH, REWRITE_MAX_CHARS)
        if not out or len(out) > limit:
            return query
        return out

    # ── Stage 2+3: Hybrid retrieval (vector + BM25) merged with RRF ──
    # ponytail: lru_cache on a method also keys on `self`, so this instance is
    # kept alive for the process lifetime. Fine here — the app builds exactly one
    # RAG and holds it anyway. Swap for an explicit dict if that ever changes.
    @lru_cache(maxsize=CACHE_SIZE)
    def _embed_query(self, query: str) -> np.ndarray:
        """Embed + normalise a query. Cached: the same question skips the model.

        Safe to cache because the result depends only on the query text and the
        embedding model — never on the documents. Re-ingesting cannot stale it.
        """
        q = self._embedder.encode(query)
        return q / np.linalg.norm(q)

    def _vector_search(self, query: str, k: int) -> list[dict]:
        q = self._embed_query(query)
        scores = self._matrix @ q                      # cosine similarity
        top = np.argsort(scores)[::-1][:k]
        return [{**self._chunks[i], "score": float(scores[i])} for i in top]

    def _bm25_search(self, query: str, k: int) -> list[dict]:
        scores = self._bm25.get_scores(query.lower().split())
        top = np.argsort(scores)[::-1][:k]
        return [{**self._chunks[i], "score": float(scores[i])}
                for i in top if scores[i] > 0]

    def hybrid_search(self, query: str) -> list[dict]:
        vec = self._vector_search(query, TOP_K_HYBRID)
        kw = self._bm25_search(query, TOP_K_HYBRID)
        merged = reciprocal_rank_fusion(vec, kw)
        return merged[:TOP_K_HYBRID], vec, kw

    # ── Stage 4: Local cross-encoder rerank ─────────────────────────
    def rerank(self, query: str, results: list[dict]) -> list[dict]:
        # Speed switch off → skip the cross-encoder, trust the RRF order.
        if not USE_RERANK or self._reranker is None:
            return results[:TOP_K_RERANK]
        pairs = [(query, r["text"]) for r in results]
        scores = self._reranker.predict(pairs)
        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)
        return sorted(results, key=lambda r: r["rerank_score"], reverse=True)[:TOP_K_RERANK]

# ── Stage 5: Generation with citations (local Ollama) ───────────
    def generate(self, query: str, chunks: list[dict]) -> str:
        # Hard refuse when retrieval is too weak to ground an answer. Runs BEFORE
        # the LLM: if reranking ran and even the best chunk falls below the floor,
        # nothing retrieved is relevant — return the same "not found" message the
        # prompt asks for, but deterministically instead of trusting the model.
        if not chunks or (
            "rerank_score" in chunks[0]
            and chunks[0]["rerank_score"] < REFUSE_BELOW_RERANK
        ):
            return "I could not find that in the documents."

        # Filter to most relevant chunks that clearly contain answer info
        filtered_chunks = []
        for i, chunk in enumerate(chunks):
            # Only include chunks that are clearly relevant.
            # .get() keeps this safe when reranking is off (no rerank_score key).
            if i < 3 or chunk.get('rerank_score', 0) > -1:
                filtered_chunks.append(chunk)
        
        # Format chunks with clear markers
        context_parts = []
        for i, c in enumerate(filtered_chunks):
            context_parts.append(f"[SOURCE {i+1}] (from {c['source']}):\n{c['text']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Keep this SIMPLE. Small local models follow short, plain instructions
        # far better than long rule-lists — an over-constrained prompt makes them
        # parrot the template ("[SOURCE N] → EXACT text") or refuse to answer.
        prompt = (
            "Answer the question using only the sources below.\n"
            "Cite sources inline like [Source 1].\n"
            "If the answer is not in the sources, say "
            "\"I could not find that in the documents.\"\n\n"
            f"Question: {query}\n\n"
            f"Sources:\n{context}\n\n"
            "Answer:"
        )
        
        # temperature=0 → deterministic, greedy decoding. For grounded factual
        # RAG we do NOT want creativity: the answer must come straight from the
        # sources. Non-zero temperature made the model occasionally "wander" and
        # claim it couldn't find facts that were right there in the context.
        resp = self._ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        return resp["message"]["content"]

    # ── Structured pipeline (for the web UI) ────────────────────────
    def query_structured(self, query: str) -> dict:
        """Run the full pipeline and RETURN every stage as data (no printing)."""
        # Retrieval searches the CLEANED question; generation answers the one the
        # user actually typed, so the reply matches what they asked.
        search_query = self.rewrite_query(query)
        merged, vec, kw = self.hybrid_search(search_query)
        reranked = self.rerank(search_query, merged)
        answer = self.generate(query, reranked)

        def slim(rows, score_key):
            return [
                {"source": r["source"],
                 "score": round(r.get(score_key, 0), 3),
                 "preview": r["text"].strip().replace("\n", " ")[:110]}
                for r in rows
            ]

        return {
            "query": query,
            # None when the rewrite changed nothing — lets the UI show it only
            # when there is actually something to show.
            "rewritten_query": search_query if search_query != query else None,
            "vector": slim(vec[:3], "score"),
            "bm25": slim(kw[:3], "score"),
            "reranked": slim(reranked, "rerank_score"),
            "answer": answer,
        }

    # ── Full pipeline with visible stage-by-stage output ────────────
    def answer(self, query: str, verify: bool = True):
        print("=" * 74)
        print(f"QUESTION: {query}")
        print("=" * 74)

        search_query = self.rewrite_query(query)
        if search_query != query:
            print(f"\n[Stage 0] Query rewritten for search:\n   {search_query!r}")

        merged, vec, kw = self.hybrid_search(search_query)

        print(f"\n[Stage 2] Vector search top 3 (by meaning):")
        for r in vec[:3]:
            print(f"   {r['score']:.3f}  {r['source']:<32} {r['text'][:55]!r}")

        print(f"\n[Stage 2] BM25 keyword search top 3 (by exact words):")
        for r in kw[:3]:
            print(f"   {r['score']:.2f}   {r['source']:<32} {r['text'][:55]!r}")

        print(f"\n[Stage 3] RRF-merged top 3 (both retrievers combined):")
        for r in merged[:3]:
            print(f"   rrf={r['rrf_score']:.4f}  {r['source']:<32} {r['text'][:50]!r}")

        reranked = self.rerank(search_query, merged)
        stage4_label = "After local rerank" if USE_RERANK else "Top RRF hits (rerank OFF)"
        print(f"\n[Stage 4] {stage4_label} — top {TOP_K_RERANK} sent to the LLM:")
        for i, r in enumerate(reranked, 1):
            score = f"{r['rerank_score']:+.2f}" if 'rerank_score' in r else f"rrf={r.get('rrf_score', 0):.4f}"
            print(f"   [Source {i}] {score}  {r['source']}")

        print(f"\n[Stage 5] Generated answer (local Ollama · {OLLAMA_MODEL}):\n")
        answer = self.generate(query, reranked)
        print(textwrap.indent(textwrap.fill(answer, 74), "   "))

        if verify:
            print(f"\n  ── Faithfulness Check ──")
            result = verify_faithfulness(answer, reranked)
            label = "✓ FAITHFUL" if result["is_faithful"] else "✗ UNFAITHUL"
            print(f"  {label}  score={result['faithfulness_score']:.2f}")
            for issue in result["issues"]:
                print(f"  ⚠  {issue}")

        print()


SAMPLE_QUESTIONS = [
    "How much is the home office stipend and when can I use it?",
    "What do I do when a SEV-1 incident happens?",             # keyword-heavy
    "Can I expense a business class flight to Tokyo?",          # cross-section reasoning
    "How quickly must reviewers respond to a pull request?",
    "Can I claim both the internet reimbursement and a co-working membership?",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid RAG demo")
    parser.add_argument("--eval", action="store_true", help="run full eval suite instead of demo questions")
    parser.add_argument("--no-verify", action="store_true", help="skip faithfulness verification")
    args = parser.parse_args()

    rag = LocalHybridRAG()
    rag.ingest()

    if args.eval:
        from eval_rag import full_report
        full_report(rag)
    else:
        for q in SAMPLE_QUESTIONS:
            rag.answer(q, verify=not args.no_verify)
