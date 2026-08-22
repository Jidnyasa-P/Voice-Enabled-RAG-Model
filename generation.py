"""
Person 3 — Generation stage.

Reuses Person 1's `_call_tool` helper from guardrails.py instead of adding
a second LLM client/provider to the project. This means generation uses
the exact same model (sarvam-105b), auth (SARVAM_API_KEY), timeout, and
forced-structured-JSON mechanism (tool calling) as the guardrail checks —
one LLM client configuration for the whole pipeline, and one less thing to
keep in sync across teammates' code.

Forces structured JSON output (answer + sources) so this needs no
prose-regexing, and so Person 1's check_grounded() and the harness's
logging can rely on a stable shape.
"""

import guardrails

RAG_SYSTEM_PROMPT = """You are a question-answering assistant. Answer the user's
QUESTION using ONLY the information in the CONTEXT passages below. Do not
use outside knowledge. If the CONTEXT does not contain enough information
to answer, say so honestly instead of guessing. Answer in the same
language as the question."""

RAG_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The answer to the question, in the same language as the question.",
        },
        "sources": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "1-based indices of the context passages actually used to answer.",
        },
    },
    "required": ["answer", "sources"],
}


def _build_user_content(question: str, retrieved_chunks: list) -> str:
    context = "\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(retrieved_chunks))
    return f"QUESTION: {question}\n\nCONTEXT:\n{context}"


def generate_answer(question: str, retrieved_chunks: list) -> dict:
    """
    Returns {"answer": str, "sources": [int, ...]}.

    Raises on malformed/unparseable output so the harness's with_retry can
    catch it and retry with a stricter reminder, per the playbook's error
    recovery plan (`_call_tool` itself already retries transient network/
    timeout errors; this raises for the harness to retry on top of that on
    genuinely malformed content).
    """
    if not retrieved_chunks:
        return {"answer": "I don't have enough grounded information to answer that.", "sources": []}

    user_content = _build_user_content(question, retrieved_chunks)
    parsed = guardrails._call_tool(
        RAG_SYSTEM_PROMPT, user_content,
        tool_name="answer_question",
        tool_description="Answer a question using only the provided context passages.",
        schema=RAG_SCHEMA,
    )

    if "answer" not in parsed:
        raise ValueError(f"Malformed generation response, missing 'answer': {parsed}")

    parsed.setdefault("sources", [])
    return parsed
