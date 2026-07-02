"""
Standalone Reciprocal Rank Fusion — no dependencies on API keys.
"""

def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    RRF score = Σ 1 / (k + rank_i)

    Args:
        vector_results: ranked list from vector search (highest score first)
        bm25_results: ranked list from BM25 search (highest score first)
        k: RRF constant (default 60)

    Returns:
        merged list sorted by RRF score descending
    """
    rrf_scores: dict[str, dict] = {}

    def _add_results(results: list[dict], rank_offset: int = 0):
        for position, item in enumerate(results):
            item_id = item["id"]
            rank = position + 1 + rank_offset
            rrf_score_val = 1.0 / (k + rank)
            if item_id not in rrf_scores:
                item_copy = dict(item)
                item_copy["rrf_score"] = rrf_score_val
                item_copy["vector_rank"] = None
                item_copy["bm25_rank"] = None
                rrf_scores[item_id] = item_copy
            else:
                rrf_scores[item_id]["rrf_score"] += rrf_score_val

            if rank_offset == 0:
                rrf_scores[item_id]["vector_rank"] = rank
            else:
                rrf_scores[item_id]["bm25_rank"] = rank

    _add_results(vector_results, rank_offset=0)
    _add_results(bm25_results, rank_offset=len(vector_results))

    return sorted(
        rrf_scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )
