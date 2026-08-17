"""
Build the static browser demo's data file.

The browser demo runs retrieval — vector, BM25, RRF, rerank, refusal — entirely
client-side. It does NOT re-chunk or re-embed the corpus: that happens here,
once, and ships as JSON. The browser only has to embed the query.

Why: embedding 23 chunks in WASM would cost seconds on every page load to
recompute something that never changes. The query is the only new text.

Run after changing demo_docs/ or any chunking setting:

    python build_static.py
"""

import json
from pathlib import Path

from local_rag import (
    LocalHybridRAG, CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL,
    REFUSE_BELOW_RERANK, TOP_K_HYBRID, TOP_K_RERANK,
)

OUT = Path(__file__).parent / "docs" / "chunks.json"
# 6 decimals is far finer than cosine similarity can resolve at this scale, and
# it roughly halves the payload.
PRECISION = 6


def build() -> dict:
    rag = LocalHybridRAG()
    chunks = rag.ingest()          # same splitter and model as the live pipeline

    # rag._matrix is already L2-normalised, so the browser can use a plain dot
    # product for cosine similarity instead of normalising 23 vectors itself.
    matrix = rag._matrix
    return {
        "model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "refuse_below_rerank": REFUSE_BELOW_RERANK,
        "top_k_hybrid": TOP_K_HYBRID,
        "top_k_rerank": TOP_K_RERANK,
        "normalized": True,
        "chunks": [
            {
                "id": c["id"],
                "source": c["source"],
                "text": c["text"],
                "vec": [round(float(v), PRECISION) for v in matrix[i]],
            }
            for i, c in enumerate(chunks)
        ],
    }


if __name__ == "__main__":
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")

    dims = len(data["chunks"][0]["vec"])
    kb = OUT.stat().st_size / 1024
    sources = sorted({c["source"] for c in data["chunks"]})
    print(f"wrote {OUT.relative_to(Path(__file__).parent)}")
    print(f"  {len(data['chunks'])} chunks x {dims} dims from {len(sources)} documents")
    print(f"  {kb:.0f} KB")
