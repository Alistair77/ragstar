"""
Faithfulness verification — checks if the LLM's cited claims are actually
supported by the source documents it was given.

This is the "LLM-as-Judge" pattern: we ask a separate LLM call to grade
each claim against the source, rather than trying to parse meaning ourselves.

Usage:
    from faithfulness import verify_faithfulness
    result = verify_faithfulness(answer, chunks)
    # result == {"is_faithful": bool, "faithfulness_score": float, "issues": [str]}
"""

import re
import ollama

OLLAMA_MODEL = "qwen3b-128k"

JUDGE_PROMPT = """You are an expert faithfulness judge. Your job is to check whether each claim in an AI-generated answer is supported by the provided source texts.

The answer cites sources like [Source 1], [Source 2], etc. For EACH cited claim:

1. Find the claim in the answer
2. Find the corresponding [Source N] text below  
3. Decide if the source EXACTLY supports the claim (no exaggeration, no addition, no contradiction)
4. If ANY claim is unsupported or exaggerated, the answer is UNFAITHFUL

Scoring rules:
- Score 1.0: Every claim is fully supported by the sources
- Score 0.7: Minor issues (small imprecision, missing nuance)  
- Score 0.3: Some claims are wrong or made up
- Score 0.0: Major hallucination or contradiction

<answer>
{answer}
</answer>

<sources>
{sources}
</sources>

Respond with EXACTLY this JSON format (no other text):
{{"faithfulness_score": <0.0-1.0>, "issues": ["list of each unsupported claim, or empty if fully faithful"]}}"""


def _format_sources(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks):
        parts.append(f"[Source {i+1}] (from: {c['source']})\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def _extract_sources_from_answer(answer: str) -> list[int]:
    """Extract all [Source N] references from the answer text."""
    matches = re.findall(r'\[Source\s+(\d+)\]', answer, re.IGNORECASE)
    return sorted(set(int(m) for m in matches))


def verify_faithfulness(answer: str, chunks: list[dict]) -> dict:
    """
    Check if the answer's cited claims are supported by the source chunks.

    Args:
        answer: The LLM-generated answer (may contain [Source N] citations)
        chunks: The list of source chunks that were provided to the LLM

    Returns:
        dict with is_faithful, faithfulness_score, issues
    """
    if not chunks:
        return {"is_faithful": True, "faithfulness_score": 1.0, "issues": []}

    refs = _extract_sources_from_answer(answer)
    if not refs:
        return {"is_faithful": True, "faithfulness_score": 1.0,
                "issues": ["No [Source N] citations found — answer may not be grounded"]}

    invalid_refs = [n for n in refs if n < 1 or n > len(chunks)]
    if invalid_refs:
        return {"is_faithful": False, "faithfulness_score": 0.0,
                "issues": [f"Answer cites [Source {n}] which doesn't exist" for n in invalid_refs]}

    sources_text = _format_sources(chunks)

    prompt = JUDGE_PROMPT.format(answer=answer, sources=sources_text)

    try:
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp["message"]["content"]

        json_match = re.search(r'\{[^}]*"faithfulness_score"[^}]*\}', raw, re.DOTALL)
        if json_match:
            import json as json_module
            result = json_module.loads(json_match.group())
        else:
            result = {"faithfulness_score": 0.5, "issues": ["Could not parse judge response"]}

    except Exception as e:
        result = {"faithfulness_score": 0.5, "issues": [f"Judge error: {e}"]}

    result["is_faithful"] = result.get("faithfulness_score", 0) >= 0.7 and not result.get("issues")
    return result
