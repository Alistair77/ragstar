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

    assert merged_ids[0] == "a", f"Expected a first, got {merged_ids}"
    assert merged_ids[1] == "c", f"Expected c second, got {merged_ids}"

    doc_a_actual = next(m for m in merged if m["id"] == "a")
    doc_c_actual = next(m for m in merged if m["id"] == "c")
    assert doc_a_actual["rrf_score"] > doc_c_actual["rrf_score"]

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


if __name__ == "__main__":
    test_single_origin()
    test_no_overlap()
    test_rrf_merge()
    print("\nAll unit tests passed!")
