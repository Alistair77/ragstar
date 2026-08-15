"""
Unit test for RRF merge logic — no API keys required.
Uses the standalone RRF function directly.
"""

from rrf import reciprocal_rank_fusion


def test_rrf_merge():
    vector_results = [
        {"id": "a", "score": 0.9, "text": "doc a"},
        {"id": "b", "score": 0.8, "text": "doc b"},
        {"id": "c", "score": 0.7, "text": "doc c"},
        {"id": "d", "score": 0.6, "text": "doc d"},
    ]
    bm25_results = [
        {"id": "c", "score": 15.0, "text": "doc c"},
        {"id": "e", "score": 12.0, "text": "doc e"},
        {"id": "a", "score": 10.0, "text": "doc a"},
        {"id": "f", "score": 8.0, "text": "doc f"},
    ]

    merged = reciprocal_rank_fusion(vector_results, bm25_results, k=60)

    merged_ids = [m["id"] for m in merged]

    # a (vec #1, bm25 #3) and c (vec #3, bm25 #1) are symmetric, so they
    # score identically and must both outrank every single-list doc.
    assert set(merged_ids[:2]) == {"a", "c"}, f"Expected a and c on top, got {merged_ids}"

    doc_a_actual = next(m for m in merged if m["id"] == "a")
    doc_c_actual = next(m for m in merged if m["id"] == "c")
    assert doc_a_actual["rrf_score"] == doc_c_actual["rrf_score"]

    # b was vector rank #2 with no bm25 hit; e was bm25 rank #2 with no
    # vector hit — with an unbiased merge they must also score identically.
    doc_b = next(m for m in merged if m["id"] == "b")
    doc_e = next(m for m in merged if m["id"] == "e")
    assert doc_b["rrf_score"] == doc_e["rrf_score"], (
        "BM25-only and vector-only docs at the same rank must score equally"
    )

    print("RRF scores:")
    for m in merged:
        print(f"  {m['id']}: rrf={m['rrf_score']:.4f} "
              f"(vec_rank={m['vector_rank']}, bm25_rank={m['bm25_rank']})")

    print("test_rrf_merge PASSED")


def test_single_origin():
    """When a doc appears in only one retriever, it still gets a non-zero score."""
    vector_results = [
        {"id": "a", "score": 0.9, "text": "doc a"},
    ]
    bm25_results: list[dict] = []

    merged = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
    assert len(merged) == 1
    assert merged[0]["id"] == "a"
    assert merged[0]["rrf_score"] > 0
    print("test_single_origin PASSED")


def test_no_overlap():
    """When retrievers return disjoint sets, both appear."""
    vector_results = [
        {"id": "a", "score": 0.9, "text": "doc a"},
    ]
    bm25_results = [
        {"id": "b", "score": 10.0, "text": "doc b"},
    ]

    merged = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
    assert len(merged) == 2
    print("test_no_overlap PASSED")


def test_refuses_when_retrieval_is_weak():
    """A below-floor top chunk must refuse WITHOUT calling the LLM.

    Calls LocalHybridRAG.generate unbound, with a bare object as self, so the test
    needs no models, no Ollama, and no ingestion. If generate() ever tried to hit
    the LLM on this input, the missing _ollama attribute would raise instead.
    """
    from local_rag import LocalHybridRAG, REFUSE_BELOW_RERANK

    REFUSAL = "I could not find that in the documents."
    dummy = object.__new__(LocalHybridRAG)          # no __init__ → no models loaded

    weak = [{"text": "irrelevant", "source": "x.md",
             "rerank_score": REFUSE_BELOW_RERANK - 1}]
    assert LocalHybridRAG.generate(dummy, "q", weak).strip() == REFUSAL
    assert LocalHybridRAG.generate(dummy, "q", []).strip() == REFUSAL

    # Chunks with no rerank_score at all (reranking off) must NOT be refused here —
    # the guard only fires on a real, below-floor score.
    no_score = [{"text": "something", "source": "x.md"}]
    try:
        LocalHybridRAG.generate(dummy, "q", no_score)
    except AttributeError:
        pass          # expected: it got past the guard and reached for _ollama
    else:
        raise AssertionError("expected generate() to proceed past the refusal guard")

    print("test_refuses_when_retrieval_is_weak PASSED")


def test_query_rewrite_falls_back_safely():
    """A bad rewrite must never replace the user's question.

    Drives rewrite_query with a fake Ollama client, so no model, no server and no
    ingestion are needed. Covers every path that must fall back to the original.
    """
    from local_rag import LocalHybridRAG, REWRITE_MAX_GROWTH, REWRITE_MAX_CHARS

    rag = object.__new__(LocalHybridRAG)        # no __init__ → no models loaded
    q = "wat is teh PTO polcy"

    class Fake:
        def __init__(self, reply): self.reply = reply
        def chat(self, *a, **k):
            if isinstance(self.reply, Exception):
                raise self.reply
            return {"message": {"content": self.reply}}

    def swap(reply):
        """Point at a new fake client. rewrite_query is lru_cached, so the cache
        must be dropped too — otherwise we would re-read the previous reply."""
        rag._ollama = Fake(reply)
        LocalHybridRAG.rewrite_query.cache_clear()

    # An empty question must not even reach the LLM (this client would explode).
    swap(RuntimeError("must not be called"))
    assert rag.rewrite_query("") == ""

    # Ollama down / model missing → search as typed.
    assert rag.rewrite_query(q) == q

    # Blank reply → keep the original.
    swap("   \n  ")
    assert rag.rewrite_query(q) == q

    # Model explained itself instead of rewriting → too long → keep the original.
    # Must beat BOTH ceilings: the 3x multiple and the absolute character floor.
    swap("x" * (max(len(q) * REWRITE_MAX_GROWTH, REWRITE_MAX_CHARS) + 1))
    assert rag.rewrite_query(q) == q

    # A short query is allowed to grow past 3x, up to the absolute ceiling —
    # this is the abbreviation case the multiplier alone used to reject.
    short = "PTO polcy"
    swap("What is the paid time off policy?")
    assert rag.rewrite_query(short) == "What is the paid time off policy?"

    # Good rewrite → used, surrounding quotes stripped.
    swap('"What is the paid time off policy?"')
    assert rag.rewrite_query(q) == "What is the paid time off policy?"

    # A leading reasoning block is discarded, not searched for.
    swap("<think>hmm</think>\nWhat is the paid time off policy?")
    assert rag.rewrite_query(q) == "What is the paid time off policy?"

    LocalHybridRAG.rewrite_query.cache_clear()     # leave no state for other tests
    print("test_query_rewrite_falls_back_safely PASSED")


def test_cache_skips_repeat_work():
    """Asking the same question twice must not call the LLM twice."""
    from local_rag import LocalHybridRAG

    rag = object.__new__(LocalHybridRAG)
    calls = []

    class Counting:
        def chat(self, *a, **k):
            calls.append(1)
            return {"message": {"content": "What is the paid time off policy?"}}

    LocalHybridRAG.rewrite_query.cache_clear()
    rag._ollama = Counting()

    first = rag.rewrite_query("PTO polcy")
    second = rag.rewrite_query("PTO polcy")

    assert first == second == "What is the paid time off policy?"
    assert len(calls) == 1, f"cache miss: expected 1 LLM call, got {len(calls)}"

    # A different question is a different key — it must still reach the LLM.
    rag.rewrite_query("holiday polcy")
    assert len(calls) == 2, f"expected 2 LLM calls, got {len(calls)}"

    LocalHybridRAG.rewrite_query.cache_clear()
    print("test_cache_skips_repeat_work PASSED")


if __name__ == "__main__":
    test_single_origin()
    test_no_overlap()
    test_rrf_merge()
    test_refuses_when_retrieval_is_weak()
    test_query_rewrite_falls_back_safely()
    test_cache_skips_repeat_work()
    print("\nAll unit tests passed!")
