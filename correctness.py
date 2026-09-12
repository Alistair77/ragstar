"""
Correctness verification — checks if the LLM's answer actually contains the
right information, not just whether its citations are supported.

Faithfulness (faithfulness.py) asks "is every cited claim backed by a
source?" — an answer can be 100% faithful and still be wrong, e.g. citing a
real source but pulling the wrong number from it. This asks the other
question: "does the answer state the expected fact?"

Usage:
    from correctness import verify_correctness
    result = verify_correctness(answer, expected_answer)
    # result == {"is_correct": bool, "correctness_score": float, "reasoning": str}
"""

import re

import ollama

OLLAMA_MODEL = "qwen2.5:3b"

# Asking qwen2.5:3b to grade a whole multi-sentence answer against an
# expected fact and return JSON+reasoning in one call was measured to fail
# badly: it scored a paragraph that opened with "the learning budget is
# $2,000" as MISSING the fact "$2,000", and separately echoed the prompt's
# own JSON-field instruction text ("one short sentence") back as its
# "reasoning" instead of writing one. Splitting the answer into single
# sentences and asking a yes/no question per sentence measured 4/5 correct
# on the same cases the whole-paragraph version got 2/4 on (see git log for
# the failed attempts). Small models do far better at "does THIS SHORT
# SENTENCE contain X" than "does this paragraph contain X, and explain why".
SENTENCE_PROMPT = """Find the exact meaning of "{expected}" inside this sentence:

"{sentence}"

Reply with exactly one word: FOUND or MISSING."""


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _sentence_states_fact(sentence: str, expected_answer: str) -> bool:
    prompt = SENTENCE_PROMPT.format(expected=expected_answer, sentence=sentence)
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return resp["message"]["content"].strip().upper().startswith("FOUND")


def verify_correctness(answer: str, expected_answer: str) -> dict:
    """Check each sentence of `answer` for the fact in `expected_answer`.

    Returns dict with is_correct, correctness_score, reasoning. The
    reasoning is built from the matching (or non-matching) sentence rather
    than asked of the model — a small model free-texting its own
    explanation was measured to be unreliable, but "which sentence said it"
    only requires one bit of judgment per sentence.
    """
    if not expected_answer:
        return {"is_correct": True, "correctness_score": 1.0,
                "reasoning": "No expected answer given"}

    try:
        for sentence in _split_sentences(answer):
            if _sentence_states_fact(sentence, expected_answer):
                return {"is_correct": True, "correctness_score": 1.0,
                        "reasoning": f'Stated in: "{sentence}"'}
    except Exception as e:
        return {"is_correct": False, "correctness_score": 0.0,
                "reasoning": f"Judge error: {e}"}

    return {"is_correct": False, "correctness_score": 0.0,
            "reasoning": f'No sentence stated "{expected_answer}"'}
