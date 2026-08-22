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
from sarvamai import SarvamAI

API_KEY = os.environ.get("SARVAM_API_KEY", "PASTE_YOUR_API_KEY_HERE")
_client = SarvamAI(api_subscription_key=API_KEY)

MODEL = "sarvam-105b"  # only supported Sarvam chat model as of Aug 2026
                        # (sarvam-30b and sarvam-m are deprecated)


def _call_json(system_prompt: str, user_content: str) -> dict:
    """
    Shared helper: calls sarvam-105b, forces JSON output, disables
    reasoning (reasoning_effort=None) since these are fast classification
    calls, not tasks that need step-by-step thinking — this keeps latency
    down, which matters for your overall pipeline budget.
    """
    response = _client.chat.completions(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        reasoning_effort=None,
        temperature=0.1,  # low temperature -> consistent classification
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


OFF_TOPIC_SYSTEM_PROMPT = """You are a strict binary classifier for a question-answering
system. The system's knowledge base is a large multilingual
corpus of general factual/informational passages (similar to
web search results) covering topics like history, science,
geography, definitions, how-to facts, and general knowledge.

Given the user's query, decide if it is the kind of
factual/informational question this system could plausibly
answer using such a corpus, OR if it is off-topic (e.g. personal
conversation, requests for opinions on the assistant itself,
code generation, creative writing, math homework, or anything
unrelated to factual lookup).

Respond with ONLY valid JSON, no other text:
{"in_domain": true or false, "reason": "<one short sentence>"}"""

UNSAFE_SYSTEM_PROMPT = """You are a safety classifier. Given the user query,
determine if it requests or contains: harassment, hate speech,
self-harm content, illegal activity instructions, sexual content
involving minors, or other clearly unsafe material.

Respond with ONLY valid JSON:
{"unsafe": true or false, "category": "<short category or null>"}"""


def check_off_topic(query: str) -> dict:
    """Returns {"in_domain": bool, "reason": str}"""
    return _call_json(OFF_TOPIC_SYSTEM_PROMPT, query)


def check_unsafe(query: str) -> dict:
    """Returns {"unsafe": bool, "category": str | None}"""
    return _call_json(UNSAFE_SYSTEM_PROMPT, query)


def check_retrieval_confidence(top_score: float, threshold: float = 0.6) -> bool:
    """
    Returns True if the top retrieved chunk's similarity score clears
    the confidence bar. No AI call needed — just a threshold check.
    Tune `threshold` once Person 2's retrieval is wired in: test a few
    values against real queries (0.55-0.65 is a reasonable starting range
    for cosine similarity, but depends on the embedding model they pick).
    """
    return top_score >= threshold


def validate_query(query: str) -> dict:
    """
    Convenience wrapper Person 3 can call as a single input-guardrail
    step: runs both checks and returns whether the query should proceed.
    """
    off_topic = check_off_topic(query)
    unsafe = check_unsafe(query)

    if unsafe["unsafe"]:
        return {"ok": False, "reason": f"unsafe content ({unsafe['category']})"}
    if not off_topic["in_domain"]:
        return {"ok": False, "reason": off_topic["reason"]}
    return {"ok": True, "reason": None}
