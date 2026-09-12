"""
RAG Evaluation Framework — measures retrieval quality and answer faithfulness.

Four metrics:
  - hit_rate@k  : was the right chunk in the top k?
  - MRR@k       : rank of the first relevant chunk (mean reciprocal rank)
  - faithfulness: does the answer's cited claims match the source?
  - refusal     : does it decline exactly when it should? (false / missed / hedged)

Usage:
    python eval_rag.py
"""

import re
import json
from dataclasses import dataclass, field

import ollama

from local_rag import LocalHybridRAG, REFUSAL_MESSAGE, REFUSE_BELOW_RERANK
from faithfulness import verify_faithfulness
from correctness import verify_correctness


@dataclass
class EvalCase:
    question: str
    expected_phrase: str          # a snippet that MUST appear in the retrieved chunk
    expected_source: str          # which doc it should come from
    difficulty: str = "medium"    # easy / medium / hard
    # True for questions the documents do NOT answer. The system passes only by
    # refusing; expected_phrase, expected_source and expected_answer are left empty.
    should_refuse: bool = False
    # The correct answer in plain language, for the LLM-judge correctness check.
    # Distinct from expected_phrase: that's a retrieval check (is this text in
    # the chunk?); this is an answer check (does the generated answer say the
    # right thing, in whatever words it picks?).
    expected_answer: str = ""


GOLDEN_DATASET: list[EvalCase] = [
    EvalCase(
        question="How much is the home office stipend and when can I use it?",
        expected_phrase="$1,500",
        expected_source="employee_handbook.md",
        difficulty="easy",
        expected_answer="A one-time $1,500 home office setup stipend.",
    ),
    EvalCase(
        question="What do I do when a SEV-1 incident happens?",
        expected_phrase="page the Incident Commander and the Head of Security",
        expected_source="security_incident_policy.md",
        difficulty="easy",
        expected_answer="Page the Incident Commander and the Head of Security.",
    ),
    EvalCase(
        question="Can I expense a business class flight to Tokyo?",
        expected_phrase="Business class requires VP approval",
        expected_source="travel_and_expense_policy.md",
        difficulty="medium",
        expected_answer="Only with VP approval; otherwise no.",
    ),
    EvalCase(
        question="How quickly must reviewers respond to a pull request?",
        expected_phrase="within one business day",
        expected_source="engineering_onboarding.md",
        difficulty="easy",
        expected_answer="Within one business day.",
    ),
    EvalCase(
        question="Can I claim both internet reimbursement and a co-working membership?",
        expected_phrase="cannot claim both",
        expected_source="employee_handbook.md",
        difficulty="easy",
        expected_answer="No, you cannot claim both at the same time.",
    ),
    EvalCase(
        question="What is the learning and development budget?",
        expected_phrase="$2,000",
        expected_source="employee_handbook.md",
        difficulty="medium",
        expected_answer="$2,000.",
    ),
    EvalCase(
        question="How many weeks of parental leave do new parents get?",
        expected_phrase="16 weeks",
        expected_source="employee_handbook.md",
        difficulty="medium",
        expected_answer="16 weeks.",
    ),
    EvalCase(
        question="What happens if customer personal data is exposed?",
        expected_phrase="notify affected customers within 72 hours",
        expected_source="security_incident_policy.md",
        difficulty="hard",
        expected_answer="Affected customers must be notified within 72 hours.",
    ),
    EvalCase(
        question="Can I expense alcohol on a solo business trip?",
        expected_phrase="Alcohol is reimbursable only during team events and client dinners",
        expected_source="travel_and_expense_policy.md",
        difficulty="hard",
        expected_answer="No — alcohol is only reimbursable at team events and client dinners, not solo trips.",
    ),
    EvalCase(
        question="What is the maximum hotel cost per night in San Francisco?",
        expected_phrase="$350 per night in high-cost cities",
        expected_source="travel_and_expense_policy.md",
        difficulty="hard",
        expected_answer="$350 per night.",
    ),

    # ── Should REFUSE: questions the documents do not answer ───────────────
    # Every topic below was grep-checked against demo_docs/ and has zero hits.
    # That check earned its keep: "sick days" and "parking" both looked
    # unanswerable and are not (the handbook grants unlimited sick leave, and
    # parking is mentioned), so they were left out instead of being baked in as
    # wrong ground truth.
    #
    # Difficulty here means how hard it is to REFUSE, not to answer:
    #   easy    off-topic; nothing retrieved even looks relevant
    #   medium  an HR topic the documents simply never cover
    #   hard    a near-miss where retrieval WILL surface something plausible,
    #           handing the model material to invent an answer from
    EvalCase(
        question="Who won the 2018 FIFA World Cup?",
        expected_phrase="", expected_source="",
        difficulty="easy", should_refuse=True,
    ),
    EvalCase(
        question="Which dental insurance provider does the company use?",
        expected_phrase="", expected_source="",
        difficulty="medium", should_refuse=True,
    ),
    EvalCase(
        question="What is the 401(k) employer match?",
        expected_phrase="", expected_source="",
        difficulty="medium", should_refuse=True,
    ),
    EvalCase(
        question="How much is the annual performance bonus?",
        expected_phrase="", expected_source="",
        difficulty="medium", should_refuse=True,
    ),
    EvalCase(
        question="What is the relocation allowance for new hires?",
        expected_phrase="", expected_source="",
        difficulty="medium", should_refuse=True,
    ),
    EvalCase(
        # "office" appears in the handbook's home-office section.
        question="Can I bring my dog to the office?",
        expected_phrase="", expected_source="",
        difficulty="hard", should_refuse=True,
    ),
    EvalCase(
        # BM25 matches "Tokyo" in the travel policy's list of high-cost cities.
        question="What is the wifi password in the Tokyo office?",
        expected_phrase="", expected_source="",
        difficulty="hard", should_refuse=True,
    ),
    EvalCase(
        # BM25 matches "membership" in the co-working reimbursement rule, which
        # sits right beside a dollar amount: prime material to invent from.
        question="How much is the gym membership reimbursement?",
        expected_phrase="", expected_source="",
        difficulty="hard", should_refuse=True,
    ),
]

# Split once so each evaluator only receives cases it can meaningfully score.
ANSWERABLE: list[EvalCase] = [c for c in GOLDEN_DATASET if not c.should_refuse]
UNANSWERABLE: list[EvalCase] = [c for c in GOLDEN_DATASET if c.should_refuse]


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


def evaluate_correctness(rag: LocalHybridRAG, cases: list[EvalCase]):
    """Is the answer actually right, not just faithful to its citations?

    Faithfulness only checks that cited claims match the source text — an
    answer can cite correctly and still pull the wrong number. This runs the
    real user-facing path and grades the reply against expected_answer.

    A refused or hedged reply is reported as such, not graded as "incorrect":
    those are refusal-eval failures, and double-counting them here would make
    a single bug look like two separate failures.
    """
    print(f"\n{'=' * 74}")
    print("CORRECTNESS EVALUATION  (LLM-as-Judge vs. expected answer)")
    print(f"{'=' * 74}")

    counts = {"correct": 0, "incorrect": 0, "refused": 0, "hedged": 0}
    for case in cases:
        d = rag.query_structured(case.question)
        verdict = classify_reply(d["answer"])

        if verdict in ("refused", "hedged"):
            counts[verdict] += 1
            print(f"  ~ {verdict:<9} (not graded)  {case.question[:55]}")
            continue

        result = verify_correctness(d["answer"], case.expected_answer)
        outcome = "correct" if result["is_correct"] else "incorrect"
        counts[outcome] += 1
        mark = "✓" if result["is_correct"] else "✗"
        print(f"  {mark} {outcome:<9} {case.question[:55]}")
        if not result["is_correct"]:
            print(f"       ⚠  {result['reasoning']}")

    graded = counts["correct"] + counts["incorrect"]
    accuracy = counts["correct"] / graded if graded else 0.0

    print(f"\n  ── Results ──")
    print(f"  Correct:   {counts['correct']}/{graded} graded ({accuracy:.0%})")
    print(f"  Incorrect: {counts['incorrect']}/{graded} graded")
    if counts["refused"] or counts["hedged"]:
        print(f"  Not graded — refused: {counts['refused']}, hedged: {counts['hedged']} "
              f"(see refusal eval for these)")

    return {**counts, "graded": graded, "accuracy": accuracy}


def classify_reply(answer: str) -> str:
    """Sort a reply into "refused", "hedged" or "answered".

    An exact-match check is not enough, and that is not hypothetical: the
    borderline business-class question produced "I could not find that in the
    documents. [Source 1] mentions economy class...". That is not equal to the
    refusal string, so an equality check scored it as a correct answer. It is
    neither a clean refusal nor a clean answer.

    Tolerant of case, surrounding whitespace and a dropped trailing full stop.
    """
    text = _normalize(answer).lower().rstrip(".")
    refusal = _normalize(REFUSAL_MESSAGE).lower().rstrip(".")
    if text == refusal:
        return "refused"
    if refusal in text:
        return "hedged"
    return "answered"


def evaluate_refusal(rag: LocalHybridRAG, cases: list[EvalCase]):
    """Does the system decline exactly when it should?

    Runs the real user-facing path (query_structured, query rewriting included)
    over answerable AND unanswerable questions and scores both ways to fail:

      false refusal   declined a question the documents do answer
      missed refusal  answered a question the documents do not cover
      hedge           said it could not find it, then answered anyway

    For every refusal it also reports WHO refused: the -6.0 gate, which never
    calls the LLM, or the model deciding for itself.
    """
    print(f"\n{'=' * 74}")
    print("REFUSAL EVALUATION  (should it answer at all?)")
    print(f"{'=' * 74}")

    counts = {"correct": 0, "false_refusal": 0, "missed_refusal": 0, "hedge": 0}
    for case in cases:
        d = rag.query_structured(case.question)
        verdict = classify_reply(d["answer"])
        top = d["reranked"][0]["score"] if d["reranked"] else float("-inf")
        who = "gate" if top < REFUSE_BELOW_RERANK else "model"

        if verdict == "hedged":
            outcome, mark = "hedge", "~"
        elif case.should_refuse:
            outcome, mark = ("correct", "✓") if verdict == "refused" else ("missed_refusal", "✗")
        else:
            outcome, mark = ("correct", "✓") if verdict == "answered" else ("false_refusal", "✗")
        counts[outcome] += 1

        expect = "refuse" if case.should_refuse else "answer"
        got = f"refused ({who})" if verdict == "refused" else verdict
        print(f"  {mark} expect {expect:<6}  got {got:<15} top={top:+6.2f}  {case.question[:46]}")

    unanswerable = sum(1 for c in cases if c.should_refuse)
    answerable = len(cases) - unanswerable
    print(f"\n  ── Results ──")
    print(f"  Correct:          {counts['correct']}/{len(cases)}")
    print(f"  False refusals:   {counts['false_refusal']}/{answerable}   declined something answerable")
    print(f"  Missed refusals:  {counts['missed_refusal']}/{unanswerable}   answered something uncovered")
    print(f"  Hedges:           {counts['hedge']}/{len(cases)}   refused and answered at once")

    return {**counts, "total": len(cases), "answerable": answerable, "unanswerable": unanswerable}


def full_report(rag: LocalHybridRAG | None = None):
    if rag is None:
        rag = LocalHybridRAG()
        rag.ingest()

    # Retrieval and faithfulness only make sense where an answer exists. An
    # unanswerable case has an empty expected_phrase, and "" is a substring of
    # every chunk, so passing one to hit_rate would score a guaranteed fake hit.
    ret = evaluate_retrieval(rag, ANSWERABLE, k=5)
    evaluate_retrieval_by_difficulty(rag, ANSWERABLE, k=5)
    faith = evaluate_faithfulness(rag, ANSWERABLE)
    correctness = evaluate_correctness(rag, ANSWERABLE)
    refusal = evaluate_refusal(rag, GOLDEN_DATASET)

    print(f"\n{'=' * 74}")
    print("SUMMARY")
    print(f"{'=' * 74}")
    print(f"  Hit-rate@5:       {ret['hit_rate']:.1%}")
    print(f"  MRR@5:            {ret['mrr']:.3f}")
    print(f"  Faithful:         {faith['faithfulness_rate']:.0%}")
    print(f"  Correct:          {correctness['correct']}/{correctness['graded']} graded "
          f"({correctness['accuracy']:.0%})")
    print(f"  Refusal correct:  {refusal['correct']}/{refusal['total']}")
    print(f"  False refusals:   {refusal['false_refusal']}   "
          f"Missed: {refusal['missed_refusal']}   Hedges: {refusal['hedge']}")
    print()

    return {"retrieval": ret, "faithfulness": faith, "correctness": correctness, "refusal": refusal}


if __name__ == "__main__":
    full_report()
