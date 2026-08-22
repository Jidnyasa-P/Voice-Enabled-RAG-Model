"""
Person 1 — Guardrail functions, wired to Sarvam's chat completion API
(sarvam-105b). Same API key you're already using for STT.

Setup:
    pip install sarvamai   (you already have this from the STT step)

These are the first two guardrails — off-topic and unsafe detection —
since those are needed by the 1:15 PM handoff to Person 3. The grounding
check and confidence threshold come later, once Person 2/3 have retrieval
and generation working (they need real retrieved chunks + a real answer
to check against).
"""

import os
import json
import time
from sarvamai import SarvamAI

API_KEY = os.environ.get("SARVAM_API_KEY", "PASTE_API_KEY_HERE")
_client = SarvamAI(api_subscription_key=API_KEY)

MODEL = "sarvam-105b"  # only supported Sarvam chat model as of Aug 2026
                        # (sarvam-30b and sarvam-m are deprecated)


def _call_tool(system_prompt: str, user_content: str, tool_name: str,
                tool_description: str, schema: dict, retries: int = 2) -> dict:
    """
    Shared helper: calls sarvam-105b and FORCES a structured response by
    defining a single "tool" whose parameters are our JSON schema, then
    setting tool_choice="required" so the model must call it. This is
    the reliable way to get structured output on the installed SDK
    version (0.1.30) — that version's chat.completions() does not accept
    a response_format argument (that's a newer API feature the SDK
    hasn't caught up to yet), but tool calling works today.

    reasoning_effort=None disables "thinking" mode, since these are fast
    classification calls, not tasks needing step-by-step reasoning —
    keeps latency down for your pipeline budget.

    Retries on read timeouts / transient network errors (retries=2 means
    up to 3 attempts total), since occasional slow responses are normal
    and shouldn't crash the whole guardrail check.
    """
    tools = [{
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_description,
            "parameters": schema,
        },
    }]

    last_error = None
    for attempt in range(retries + 1):
        try:
            response = _client.chat.completions(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                tools=tools,
                tool_choice="required",
                reasoning_effort=None,
                temperature=0.1,  # low temperature -> consistent classification
                request_options={"timeout_in_seconds": 30},
            )
            message = response.choices[0].message
            if message.tool_calls:
                args_json = message.tool_calls[0].function.arguments
                return json.loads(args_json)
            # Fallback: model replied in plain text instead of calling the
            # tool. Try to parse it as JSON directly.
            return json.loads(message.content)
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))  # brief backoff before retry
                continue
            raise last_error


OFF_TOPIC_SYSTEM_PROMPT = """You are a strict binary classifier for a question-answering
system. The system's knowledge base is a large multilingual
corpus of general factual/informational TEXT PASSAGES (similar to
web search results) covering topics like history, science,
geography, definitions, how-to facts, and general knowledge.

A query is IN-DOMAIN only if the answer could plausibly be found
written down inside such a passage (e.g. "what is the capital of
France", "how does photosynthesis work").

A query is OFF-TOPIC if answering it requires something other than
retrieving and reading a passage — even if it sounds "factual" on
the surface. This includes:
- Arithmetic or numeric calculation (e.g. "what's 348 times 27") —
  the answer isn't written in any passage, it must be computed.
- Code generation, creative writing, poetry, jokes.
- Personal opinions about the assistant itself, or casual conversation.
- Math/logic puzzles, unit conversions, date/time calculations.

When in doubt, ask: "would the answer to this literally appear as
text inside a knowledge-base passage, or would it need to be
computed/generated instead?" If it needs to be computed or
generated, it is OFF-TOPIC."""

UNSAFE_SYSTEM_PROMPT = """You are a safety classifier. Given the user query,
determine if it requests or contains: harassment, hate speech,
self-harm content, illegal activity instructions, sexual content
involving minors, or other clearly unsafe material."""

OFF_TOPIC_SCHEMA = {
    "type": "object",
    "properties": {
        "in_domain": {"type": "boolean", "description": "True if the query is answerable from a general factual knowledge base."},
        "reason": {"type": "string", "description": "One short sentence explaining the decision."},
    },
    "required": ["in_domain", "reason"],
}

UNSAFE_SCHEMA = {
    "type": "object",
    "properties": {
        "unsafe": {"type": "boolean", "description": "True if the query is unsafe or inappropriate."},
        "category": {"type": ["string", "null"], "description": "Short category name if unsafe, else null."},
    },
    "required": ["unsafe", "category"],
}


def check_off_topic(query: str) -> dict:
    """Returns {"in_domain": bool, "reason": str}"""
    return _call_tool(
        OFF_TOPIC_SYSTEM_PROMPT, query,
        tool_name="classify_domain",
        tool_description="Classify whether a user query is in-domain (factual/informational) or off-topic.",
        schema=OFF_TOPIC_SCHEMA,
    )


def check_unsafe(query: str) -> dict:
    """Returns {"unsafe": bool, "category": str | None}"""
    return _call_tool(
        UNSAFE_SYSTEM_PROMPT, query,
        tool_name="classify_safety",
        tool_description="Classify whether a user query is unsafe or inappropriate.",
        schema=UNSAFE_SCHEMA,
    )


def check_retrieval_confidence(top_score: float, threshold: float = 0.6) -> bool:
    """
    Returns True if the top retrieved chunk's similarity score clears
    the confidence bar. No AI call needed — just a threshold check.

    threshold=0.6 is a reasonable STARTING point for cosine similarity,
    but the right value depends entirely on Person 2's embedding model —
    different models produce different similarity score distributions.
    Once retrieval is live, ask Person 2 for the top-1 scores across
    ~10 real in-domain queries and ~10 queries you'd expect the corpus
    to have nothing good for, then pick a threshold that separates them.
    See test_guardrails.py for a synthetic version of this tuning exercise.
    """
    return top_score >= threshold


GROUNDING_SYSTEM_PROMPT = """You are a fact-checking module for a RAG (retrieval-augmented
generation) system. You will be given a QUESTION, a set of CONTEXT
passages retrieved from a knowledge base, and an ANSWER generated by
another system using that context.

Determine whether the ANSWER is fully supported by the CONTEXT —
i.e. every factual claim in the ANSWER can be traced back to
something actually stated in the CONTEXT. Flag any claim that is
NOT backed by the CONTEXT, including:
- Facts, numbers, or names not present in any passage
- Claims that go beyond what the CONTEXT actually says
- Claims that contradict the CONTEXT

Minor rephrasing or reasonable summarization of context content is
fine and should NOT be flagged. Only flag claims that introduce
information the CONTEXT does not support."""

GROUNDING_SCHEMA = {
    "type": "object",
    "properties": {
        "grounded": {"type": "boolean", "description": "True only if every claim in the answer is supported by the context."},
        "unsupported_claims": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of specific claims from the answer that are NOT supported by the context. Empty list if grounded is true.",
        },
    },
    "required": ["grounded", "unsupported_claims"],
}


def check_grounded(question: str, context_chunks: list, answer: str) -> dict:
    """
    Returns {"grounded": bool, "unsupported_claims": [str, ...]}

    context_chunks: list of retrieved chunk text strings (or dicts with a
    "text" key — both accepted below), exactly what Person 2's retrieve()
    returns for this query and Person 3's generation used to produce
    `answer`.

    This is the single most convincing guardrail for judges — it's what
    lets the system say "I don't have enough grounded information" instead
    of confidently hallucinating.
    """
    texts = [c["text"] if isinstance(c, dict) else c for c in context_chunks]
    context_block = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))

    user_content = (
        f"QUESTION: {question}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"ANSWER: {answer}"
    )

    return _call_tool(
        GROUNDING_SYSTEM_PROMPT, user_content,
        tool_name="check_grounding",
        tool_description="Check whether an answer is fully supported by the given context passages.",
        schema=GROUNDING_SCHEMA,
    )


def validate_query(query: str) -> dict:
    """
    Convenience wrapper Person 3 can call as a single input-guardrail
    step: runs both input checks and returns whether the query should
    proceed to retrieval.
    """
    off_topic = check_off_topic(query)
    unsafe = check_unsafe(query)

    if unsafe["unsafe"]:
        return {"ok": False, "reason": f"unsafe content ({unsafe['category']})"}
    if not off_topic["in_domain"]:
        return {"ok": False, "reason": off_topic["reason"]}
    return {"ok": True, "reason": None}


FALLBACK_MESSAGE = "I don't have enough grounded information to answer that confidently."


def validate_answer(question: str, context_chunks: list, answer: str,
                     top_score: float, confidence_threshold: float = 0.6) -> dict:
    """
    Convenience wrapper Person 3 can call as a single output-guardrail
    step, after generation: checks retrieval confidence AND grounding
    together, and returns the answer to actually show the user (either
    the real answer, or the safe fallback message).

    Returns {"ok": bool, "answer": str, "reason": str | None}
    """
    if not check_retrieval_confidence(top_score, confidence_threshold):
        return {"ok": False, "answer": FALLBACK_MESSAGE, "reason": "low retrieval confidence"}

    grounding = check_grounded(question, context_chunks, answer)
    if not grounding["grounded"]:
        return {
            "ok": False,
            "answer": FALLBACK_MESSAGE,
            "reason": f"ungrounded claims: {grounding['unsupported_claims']}",
        }

    return {"ok": True, "answer": answer, "reason": None}
