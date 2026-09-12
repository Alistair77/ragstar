"""
calibrate.py — is REFUSE_BELOW_RERANK -6.0 still right for THIS corpus?

That constant in local_rag.py was picked by eyeballing scores on the demo
docs. Swap in different documents and the scale can shift entirely — a
cross-encoder's raw logits aren't calibrated to a fixed range. This script
re-measures it: run every golden question through the real retrieval path
(query rewrite + hybrid search + rerank), print the rerank-score
distribution for answerable vs. unanswerable questions, and report whether
a threshold can actually separate them.

It does NOT just print a midpoint and call it done. The refusal eval
(eval_rag.py) found the two ranges overlap on near-misses — a gym-
membership question that isn't covered by the docs scores higher (-0.38)
than a business-class question that genuinely is answered (-4.75). When
that overlap exists, no floor is "correct"; this script says so explicitly
instead of hiding it behind an average.

Usage:
    python calibrate.py
"""

import statistics

from local_rag import LocalHybridRAG, REFUSE_BELOW_RERANK
from eval_rag import ANSWERABLE, UNANSWERABLE, EvalCase


def top_rerank_score(rag: LocalHybridRAG, case: EvalCase) -> float:
    """Best rerank score for a question, via the same path used at query time."""
    search_query = rag.rewrite_query(case.question)
    merged, _, _, _ = rag.retrieve(search_query)
    reranked = rag.rerank(search_query, merged)
    return reranked[0]["rerank_score"] if reranked else float("-inf")


def find_best_threshold(answerable: list[float], unanswerable: list[float]):
    """The floor that misclassifies the fewest questions, and how many that is.

    Not "the correct threshold" — when the two lists overlap, every
    candidate misclassifies something. This picks the least-bad one so the
    overlap can be sized concretely instead of left as a vague warning.
    """
    best_threshold, best_errors = None, None
    for t in sorted(set(answerable) | set(unanswerable)):
        false_refusals = sum(1 for s in answerable if s < t)
        missed_refusals = sum(1 for s in unanswerable if s >= t)
        errors = false_refusals + missed_refusals
        if best_errors is None or errors < best_errors:
            best_threshold, best_errors = t, errors
    return best_threshold, best_errors


def overlap_range(answerable: list[float], unanswerable: list[float]):
    """(lo, hi) of the overlapping band, or None if the groups don't overlap."""
    lo, hi = max(unanswerable), min(answerable)
    return (lo, hi) if lo >= hi else None


def _describe(label: str, scores: list[float]) -> None:
    print(f"  {label:<13} n={len(scores):<3} "
          f"min={min(scores):+7.2f}  max={max(scores):+7.2f}  "
          f"mean={statistics.mean(scores):+7.2f}  median={statistics.median(scores):+7.2f}")


def calibrate(rag: LocalHybridRAG | None = None) -> dict:
    if rag is None:
        rag = LocalHybridRAG()
        rag.ingest()

    answerable = [top_rerank_score(rag, c) for c in ANSWERABLE]
    unanswerable = [top_rerank_score(rag, c) for c in UNANSWERABLE]

    print("=" * 74)
    print("RERANK SCORE CALIBRATION")
    print("=" * 74)
    print(f"\n  Current REFUSE_BELOW_RERANK = {REFUSE_BELOW_RERANK}\n")
    _describe("Answerable", answerable)
    _describe("Unanswerable", unanswerable)

    overlap = overlap_range(answerable, unanswerable)
    print()
    if overlap:
        lo, hi = overlap
        print(f"  OVERLAP: the highest unanswerable score ({lo:+.2f}) is ABOVE")
        print(f"  the lowest answerable score ({hi:+.2f}). No single threshold")
        print(f"  separates these groups — a floor that rejects the overlap band")
        print(f"  ({hi:+.2f}..{lo:+.2f}) rejects real answers too; a floor that keeps")
        print(f"  them lets a near-miss unanswerable question through the gate.")

        threshold, errors = find_best_threshold(answerable, unanswerable)
        print(f"\n  Least-bad floor measured here: {threshold:+.2f} "
              f"({errors} question(s) still misclassified at that floor).")
        print(f"  This is not a safe number, just the smallest error achievable by")
        print(f"  a single threshold. The gate is not the last line of defence —")
        print(f"  see the refusal eval in eval_rag.py for what catches the rest.")
    else:
        lo, hi = max(unanswerable), min(answerable)
        recommended = (lo + hi) / 2
        print(f"  NO OVERLAP. Every answerable score sits above every unanswerable one.")
        print(f"  Empty gap: {lo:+.2f} .. {hi:+.2f}")
        print(f"  Recommended floor (middle of the gap): {recommended:+.2f}")

    return {"answerable": answerable, "unanswerable": unanswerable, "overlap": overlap}


if __name__ == "__main__":
    calibrate()
