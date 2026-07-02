"""
Test script — 3 sample questions against the toy document set.
Runs all 5 stages and prints results.
"""

from pathlib import Path
import sys

# Ensure we can import from this directory
sys.path.insert(0, str(Path(__file__).parent))

from stage1_ingestion import ingest
from core import HybridRAG

SAMPLE_QUESTIONS = [
    "What is the remote work policy for internet stipends?",
    "How does the company handle SEV-1 security incidents?",
    "What benefits are available to new employees?",
]


def run_stages():
    print("\n" + "=" * 70)
    print("HYBRID SEARCH RAG — FULL PIPELINE TEST")
    print("=" * 70)

    # Stage 1: Ingestion
    print("\n>>> STAGE 1: Loading documents and ingesting to Pinecone...")
    chunks = ingest()

    # Stage 2-3: BM25 + Hybrid RRF
    print("\n>>> STAGE 2 & 3: Building BM25 index and testing hybrid search...")
    rag = HybridRAG()
    rag.build_bm25_index(chunks)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"\n{'─' * 70}")
        print(f"QUESTION {i}: {question}")
        print(f"{'─' * 70}")

        # Stage 3: Hybrid search
        print("\n[Stage 3] Hybrid search (vector + BM25 + RRF)...")
        hybrid_results = rag.hybrid_search(question)
        print(f"  Retrieved {len(hybrid_results)} chunks via RRF")
        for j, r in enumerate(hybrid_results[:3]):
            print(f"  #{j+1}: [{r['source']}] score={r.get('rrf_score', 0):.4f} — "
                  f"{r['text'][:80]}...")

        # Stage 4: Cohere reranking
        print(f"\n[Stage 4] Cohere reranking (top-{len(hybrid_results)} → top-5)...")
        reranked = rag.rerank(question, hybrid_results)
        print(f"  Kept {len(reranked)} chunks after reranking")
        for j, r in enumerate(reranked):
            print(f"  #{j+1}: [{r['source']}] rerank={r.get('rerank_score', 0):.4f} — "
                  f"{r['text'][:80]}...")

        # Stage 5: Generation
        print("\n[Stage 5] Generating answer with Claude + inline citations...")
        result = rag.generate(question, reranked)

        print(f"\n  ANSWER:\n  {result['answer']}")
        print(f"\n  CITATIONS ({len(result['sources'])} sources):")
        for j, src in enumerate(result['sources']):
            print(f"    [{j+1}] {src['source']} — {src['text_preview'][:100]}...")

    print(f"\n{'=' * 70}")
    print("ALL STAGES COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_stages()
