"""
RAG Evaluation Framework — measures retrieval quality and answer faithfulness.

Three metrics:
  - hit_rate@k  : was the right chunk in the top k?
  - MRR@k       : rank of the first relevant chunk (mean reciprocal rank)
  - faithfulness: does the answer's cited claims match the source?

Usage:
    python eval_rag.py
"""

import re
import json
from dataclasses import dataclass, field

import ollama

from local_rag import LocalHybridRAG
from faithfulness import verify_faithfulness


@dataclass
class EvalCase:
    question: str
    expected_phrase: str          # a snippet that MUST appear in the retrieved chunk
    expected_source: str          # which doc it should come from
    difficulty: str = "medium"    # easy / medium / hard


GOLDEN_DATASET: list[EvalCase] = [
    EvalCase(
        question="How much is the home office stipend and when can I use it?",
        expected_phrase="$1,500",
        expected_source="employee_handbook.md",
        difficulty="easy",
    ),
    EvalCase(
        question="What do I do when a SEV-1 incident happens?",
        expected_phrase="page the Incident Commander and the Head of Security",
        expected_source="security_incident_policy.md",
        difficulty="easy",
    ),
    EvalCase(
        question="Can I expense a business class flight to Tokyo?",
        expected_phrase="Business class requires VP approval",
        expected_source="travel_and_expense_policy.md",
        difficulty="medium",
    ),
    EvalCase(
        question="How quickly must reviewers respond to a pull request?",
        expected_phrase="within one business day",
        expected_source="engineering_onboarding.md",
        difficulty="easy",
    ),
    EvalCase(
        question="Can I claim both internet reimbursement and a co-working membership?",
        expected_phrase="cannot claim both",
        expected_source="employee_handbook.md",
        difficulty="easy",
    ),
    EvalCase(
        question="What is the learning and development budget?",
        expected_phrase="$2,000",
        expected_source="employee_handbook.md",
        difficulty="medium",
    ),
    EvalCase(
        question="How many weeks of parental leave do new parents get?",
        expected_phrase="16 weeks",
        expected_source="employee_handbook.md",
        difficulty="medium",
    ),
    EvalCase(
        question="What happens if customer personal data is exposed?",
        expected_phrase="notify affected customers within 72 hours",
        expected_source="security_incident_policy.md",
        difficulty="hard",
    ),
    EvalCase(
        question="Can I expense alcohol on a solo business trip?",
        expected_phrase="Alcohol is reimbursable only during team events and client dinners",
        expected_source="travel_and_expense_policy.md",
        difficulty="hard",
    ),
    EvalCase(
        question="What is the maximum hotel cost per night in San Francisco?",
        expected_phrase="$350 per night in high-cost cities",
        expected_source="travel_and_expense_policy.md",
        difficulty="hard",
    ),
]


def _normalize(text: str) -> str:
    """Collapse whitespace so chunk-level newlines don't break substring checks."""
    return " ".join(text.split())


def hit_rate(results: list[dict], case: EvalCase) -> bool:
    norm_phrase = _normalize(case.expected_phrase).lower()
    for r in results:
        if norm_phrase in _normalize(r["text"]).lower():
            return True
    return False


def reciprocal_rank(results: list[dict], case: EvalCase) -> float:
    norm_phrase = _normalize(case.expected_phrase).lower()
    for i, r in enumerate(results):
        if norm_phrase in _normalize(r["text"]).lower():
            return 1.0 / (i + 1)
    return 0.0


def evaluate_retrieval(rag: LocalHybridRAG, cases: list[EvalCase], k: int = 5):
    print(f"\n{'=' * 74}")
    print(f"RETRIEVAL EVALUATION  (top-{k})")
    print(f"{'=' * 74}")

    total = len(cases)
    hits = 0
    rr_sum = 0.0
    results_detail = []

    for case in cases:
        merged, vec, kw = rag.hybrid_search(case.question)
        top_k = merged[:k]

        hit = hit_rate(top_k, case)
        rr = reciprocal_rank(top_k, case)

        if hit:
            hits += 1
        rr_sum += rr

        label = "✓" if hit else "✗"
        difficulty_tag = f"[{case.difficulty.upper()}]"
        print(f"  {label} {case.difficulty.upper():<6} {case.question:<65} rr={rr:.3f}")

        results_detail.append({
            "question": case.question,
            "difficulty": case.difficulty,
            "hit": hit,
            "reciprocal_rank": round(rr, 3),
            "top_source": top_k[0]["source"] if top_k else None,
        })

    hit_rate_val = hits / total
    mrr = rr_sum / total

    print(f"\n  ── Results ──")
    print(f"  Hit-rate@{k}: {hit_rate_val:.1%} ({hits}/{total})")
    print(f"  MRR@{k}:      {mrr:.3f}")

    return {"hit_rate": hit_rate_val, "mrr": mrr, "cases": results_detail}


def evaluate_retrieval_by_difficulty(rag: LocalHybridRAG, cases: list[EvalCase], k: int = 5):
    print(f"\n{'─' * 74}")
    print("BY DIFFICULTY")
    print(f"{'─' * 74}")

    for diff in ["easy", "medium", "hard"]:
        subset = [c for c in cases if c.difficulty == diff]
        if not subset:
            continue
        hits = sum(1 for c in subset if hit_rate(rag.hybrid_search(c.question)[0][:k], c))
        print(f"  {diff.upper():<8} {hits}/{len(subset)}  ({hits/len(subset):.0%})")


def evaluate_faithfulness(rag: LocalHybridRAG, cases: list[EvalCase]):
    print(f"\n{'=' * 74}")
    print("FAITHFULNESS EVALUATION  (LLM-as-Judge)")
    print(f"{'=' * 74}")

    total = 0
    faithful = 0
    score_sum = 0.0

    for case in cases:
        merged, _, _ = rag.hybrid_search(case.question)
        reranked = rag.rerank(case.question, merged)
        answer = rag.generate(case.question, reranked)

        result = verify_faithfulness(answer, reranked)
        total += 1
        if result["is_faithful"]:
            faithful += 1
        score_sum += result["faithfulness_score"]

        label = "✓" if result["is_faithful"] else "✗"
        issues = f" — {result['issues'][0]}" if result["issues"] else ""
        print(f"  {label} score={result['faithfulness_score']:.2f}{issues}")
        print(f"       Q: {case.question[:60]}")

        if not result["is_faithful"]:
            for issue in result["issues"][:2]:
                print(f"       ⚠  {issue}")

    faithfulness_rate = faithful / total if total else 0
    avg_score = score_sum / total if total else 0

    print(f"\n  ── Results ──")
    print(f"  Faithfulness rate: {faithfulness_rate:.0%} ({faithful}/{total})")
    print(f"  Average score:     {avg_score:.2f}")

    return {"faithfulness_rate": faithfulness_rate, "avg_score": avg_score}


def full_report(rag: LocalHybridRAG | None = None):
    if rag is None:
        rag = LocalHybridRAG()
        rag.ingest()

    ret = evaluate_retrieval(rag, GOLDEN_DATASET, k=5)
    evaluate_retrieval_by_difficulty(rag, GOLDEN_DATASET, k=5)
    faith = evaluate_faithfulness(rag, GOLDEN_DATASET[:5])

    print(f"\n{'=' * 74}")
    print("SUMMARY")
    print(f"{'=' * 74}")
    print(f"  Hit-rate@5: {ret['hit_rate']:.1%}")
    print(f"  MRR@5:      {ret['mrr']:.3f}")
    print(f"  Faithful:   {faith['faithfulness_rate']:.0%}")
    print()

    return {"retrieval": ret, "faithfulness": faith}


if __name__ == "__main__":
    full_report()
