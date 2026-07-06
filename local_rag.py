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

    # ── Stage 2+3: Hybrid retrieval (vector + BM25) merged with RRF ──
    def _vector_search(self, query: str, k: int) -> list[dict]:
        q = self._embedder.encode(query)
        q = q / np.linalg.norm(q)
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
        merged, vec, kw = self.hybrid_search(query)
        reranked = self.rerank(query, merged)
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

        merged, vec, kw = self.hybrid_search(query)

        print(f"\n[Stage 2] Vector search top 3 (by meaning):")
        for r in vec[:3]:
            print(f"   {r['score']:.3f}  {r['source']:<32} {r['text'][:55]!r}")

        print(f"\n[Stage 2] BM25 keyword search top 3 (by exact words):")
        for r in kw[:3]:
            print(f"   {r['score']:.2f}   {r['source']:<32} {r['text'][:55]!r}")

        print(f"\n[Stage 3] RRF-merged top 3 (both retrievers combined):")
        for r in merged[:3]:
            print(f"   rrf={r['rrf_score']:.4f}  {r['source']:<32} {r['text'][:50]!r}")

        reranked = self.rerank(query, merged)
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
