"""
One-shot runner: ingest → BM25 → test queries
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from stage1_ingestion import ingest
from core import HybridRAG
from test_rag import SAMPLE_QUESTIONS


def run():
    chunks = ingest()

    rag = HybridRAG()
    rag.build_bm25_index(chunks)

    for q in SAMPLE_QUESTIONS:
        print(f"\n>>> Q: {q}")
        result = rag.query(q)
        print(f"  A: {result['answer'][:300]}...")
        print(f"  Sources: {[s['source'] for s in result['sources']]}")


if __name__ == "__main__":
    run()
