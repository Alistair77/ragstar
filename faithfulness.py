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

JUDGE_PROMPT = """You are an expert faithfulness judge with STRICT criteria. You MUST reject ANY answer that contains hallucinated information: invented details, dates, or amounts, or claims that go beyond or contradict the sources.

For EACH factual claim in the answer:

1. Find the claim's [Source N] citation.
2. Locate the text of that [Source N] in the SOURCES below.
3. Decide: is the claim directly supported by that source? (The fact must appear in, or follow plainly from, the source text. Faithful paraphrasing is fine; invented or contradicted facts are not.)

Examples of SUPPORTED claims (VALID):
- Claim: "The home office stipend is $1,500" · Source: "...one-time home office setup stipend of $1,500" -> supported
- Claim: "Reviewers respond within one business day" · Source: "Reviewers should respond within one business day" -> supported

Examples of UNSUPPORTED claims (INVALID / HALLUCINATION):
- Claim: "The stipend is $1,500" · Source: "The home office stipend is available" -> NOT supported (amount not in source)
- Claim: "You can expense business class" · Source: "premium economy only" -> NOT supported (contradicts source)

Scoring (zero tolerance for hallucination):
- 1.0 = every cited claim is supported by its source
- 0.0 = any invented detail, contradiction, or unsupported claim

--- ANSWER TO GRADE ---
{answer}

--- SOURCES ---
{sources}

Respond with ONLY this JSON, nothing else:
{{"faithfulness_score": 1.0 or 0.0, "issues": [] or ["the specific claim that isn't supported"]}}"""


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
        # If no citations found, we can't verify faithfulness
        return {"is_faithful": False, "faithfulness_score": 0.0,
                "issues": ["No [Source N] citations found — answer may not be grounded"]}

    invalid_refs = [n for n in refs if n < 1 or n > len(chunks)]
    if invalid_refs:
        return {"is_faithful": False, "faithfulness_score": 0.0,
                "issues": [f"Answer cites [Source {n}] which doesn't exist" for n in invalid_refs]}

    sources_text = _format_sources(chunks)

    prompt = JUDGE_PROMPT.format(answer=answer, sources=sources_text)

    try:
        # temperature=0 → a deterministic judge. A grader that returns a
        # different score each run is worse than useless, so the judge must be
        # as repeatable as the generation it grades.
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        raw = resp["message"]["content"]

        # Try to extract JSON from the response - handle各种格式
        import json as json_module
        
        # First, try to find JSON in the response using regex patterns
        json_match = re.search(r'\{[\s\S]*?"faithfulness_score"[\s\S]*?\}', raw)
        if not json_match:
            # Try simpler regex
            json_match = re.search(r'\{[^\}]*"faithfulness_score"[^\}]*\}', raw)
        
        result = None
        if json_match:
            try:
                result = json_module.loads(json_match.group())
                if "faithfulness_score" in result and "issues" in result:
                    pass  # Found valid JSON
                else:
                    result = None
            except (json_module.JSONDecodeError, ValueError) as e:
                # If JSON parsing fails, result stays None
                pass
        
        # If regex didn't work, try to parse the entire response as JSON
        if not result:
            # Clean up common issues in JSON responses
            cleaned = raw.strip()
            # Remove common prefixes/suffixes
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            # Try to parse if it looks like JSON
            if cleaned.startswith('{"') and cleaned.endswith('}'):
                try:
                    result = json_module.loads(cleaned)
                except (json_module.JSONDecodeError, ValueError):
                    pass
        
        # If still no valid result, create a default failure
        if not result:
            result = {"faithfulness_score": 0.0, "issues": ["Could not parse judge response"]}

    except Exception as e:
        # On any error, treat as failure - better to flag potential hallucination
        result = {"faithfulness_score": 0.0, "issues": [f"Judge error: {e}"]}

    # Final check: is_faithful only if score is 1.0 AND no issues
    result["is_faithful"] = result.get("faithfulness_score", 0) >= 1.0 and not result.get("issues")
    return result
