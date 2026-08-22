"""
Person 1 — Guardrail functions.

Setup:
    pip install sarvamai
    export SARVAM_API_KEY="your_key"

Functions:
    check_off_topic(query) -> {in_domain: bool, reason: str}
    check_unsafe(query) -> {unsafe: bool, category: str|None}
    check_retrieval_confidence(top_score, threshold=0.6) -> bool
    check_grounded(question, context, answer) -> {grounded: bool, unsupported_claims: [...]}
    validate_query(query) -> {ok: bool, reason: str|None}
"""

import os
import json
import time
from sarvamai import SarvamAI

API_KEY = os.environ.get("SARVAM_API_KEY", "sk_qzat9w84_2x9x86OURsZY1EHOqZc2bKW4")
_client = SarvamAI(api_subscription_key=API_KEY)

MODEL = "sarvam-105b"


def _call_tool(system_prompt: str, user_content: str, tool_name: str,
                tool_description: str, schema: dict, retries: int = 2) -> dict:
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
                temperature=0.1,
                request_options={"timeout_in_seconds": 30},
            )
            message = response.choices[0].message
            if message.tool_calls:
                args_json = message.tool_calls[0].function.arguments
                return json.loads(args_json)
            return json.loads(message.content)
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
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
    return _call_tool(
        OFF_TOPIC_SYSTEM_PROMPT, query,
        tool_name="classify_domain",
        tool_description="Classify whether a user query is in-domain (factual/informational) or off-topic.",
        schema=OFF_TOPIC_SCHEMA,
    )


def check_unsafe(query: str) -> dict:
    return _call_tool(
        UNSAFE_SYSTEM_PROMPT, query,
        tool_name="classify_safety",
        tool_description="Classify whether a user query is unsafe or inappropriate.",
        schema=UNSAFE_SCHEMA,
    )


def check_retrieval_confidence(top_score: float, threshold: float = 0.6) -> bool:
    return top_score >= threshold


def validate_query(query: str) -> dict:
    off_topic = check_off_topic(query)
    unsafe = check_unsafe(query)
    if unsafe["unsafe"]:
        return {"ok": False, "reason": f"unsafe content ({unsafe['category']})"}
    if not off_topic["in_domain"]:
        return {"ok": False, "reason": off_topic["reason"]}
    return {"ok": True, "reason": None}


# ---- Output guardrail: grounding check ----

GROUNDING_SYSTEM_PROMPT = """You are a fact-checking module. You will be given a QUESTION,
a set of CONTEXT passages retrieved from a knowledge base, and
an ANSWER generated by another system using that context.

Determine whether the ANSWER is fully supported by the CONTEXT.
Flag any claim in the ANSWER that is not backed by the CONTEXT.

Respond with ONLY valid JSON:
{
  "grounded": true or false,
  "unsupported_claims": ["<claim text>", ...]
}"""

GROUNDING_SCHEMA = {
    "type": "object",
    "properties": {
        "grounded": {"type": "boolean", "description": "True if every claim in the answer is supported by the context."},
        "unsupported_claims": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of claims from the answer that are NOT supported by the context.",
        },
    },
    "required": ["grounded", "unsupported_claims"],
}


def check_grounded(question: str, context: list, answer: str) -> dict:
    context_text = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(context))
    user_content = f"QUESTION: {question}\n\nCONTEXT:\n{context_text}\n\nANSWER: {answer}"
    return _call_tool(
        GROUNDING_SYSTEM_PROMPT, user_content,
        tool_name="check_grounding",
        tool_description="Check whether an answer is fully supported by retrieved context passages.",
        schema=GROUNDING_SCHEMA,
    )
