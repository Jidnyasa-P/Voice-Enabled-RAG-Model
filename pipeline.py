"""
Person 3 — Orchestration harness.

Wires together:
  - stt.transcribe()                          (Person 1)
  - guardrails.check_off_topic/check_unsafe   (Person 1, input guardrails)
  - RetrievalEngine.retrieve()                (Person 2)
  - generation.generate_answer()              (Person 3)
  - guardrails.check_grounded()               (Person 1, output guardrail)
  - guardrails.check_retrieval_confidence()   (Person 1, confidence guardrail)

A few adjustments vs. the playbook's pseudocode were needed to match what
Person 1 and Person 2 actually shipped, not what the draft assumed:

  1. `transcribe(audio)` returns a DICT ({"transcript", "language",
     "confidence"}), not a 3-tuple — the draft's
     `state.transcript, state.language, _ = transcribe(audio)` would have
     raised a ValueError. This module unpacks it by key instead.

  2. Retrieval is a method on a loaded RetrievalEngine instance
     (`engine.retrieve(...)`), not a bare module-level `retrieve()`
     function — the models/index need to be loaded once and reused, so
     run_pipeline() takes an already-constructed `engine` rather than
     importing a free function.

  3. The confidence check now actually calls Person 1's
     `check_retrieval_confidence()` guardrail function (as their README
     specifies) instead of re-implementing the threshold comparison
     inline.

  4. run_pipeline() accepts EITHER `audio=` (the normal path) OR a
     pre-transcribed `query_text=`/`language=` pair, which skips the STT
     stage. This is what lets benchmark_pipeline.py measure guardrails +
     retrieval + generation + grounding latency across 30+ queries
     without needing 30+ real recorded audio clips — see
     benchmark_pipeline.py for how this is used and reported honestly.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from stt import transcribe
from guardrails import check_off_topic, check_unsafe, check_grounded, check_retrieval_confidence
from generation import generate_answer
from retry_utils import with_retry

# Matches guardrails.check_retrieval_confidence's own default — kept as a
# named constant here so pipeline.py and any caller can see/override it
# without reaching into guardrails' function signature.
CONFIDENCE_THRESHOLD = 0.6

TRY_AGAIN_MESSAGE = "Sorry, I couldn't process that — please try again."
NO_CONTEXT_MESSAGE = "I don't have enough grounded information to answer that."
UNGROUNDED_MESSAGE = "I don't have enough grounded information to answer that confidently."


@dataclass
class PipelineState:
    audio: Optional[bytes] = None
    transcript: str = ""
    language: str = ""
    query_ok: bool = True
    reject_reason: str = ""
    retrieved: List[Dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    sources: List[int] = field(default_factory=list)
    grounded: bool = True
    timings: Dict[str, float] = field(default_factory=dict)


def run_pipeline(
    engine,
    audio: bytes = None,
    query_text: str = None,
    language: str = None,
    top_k: int = 3,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> PipelineState:
    """
    engine: a loaded retrieval_engine.RetrievalEngine instance. Construct
        this ONCE (e.g. at app startup / benchmark startup) and pass it in
        on every call — it owns the embedding model, re-ranker, and Chroma
        connection, all of which are expensive to (re)load per query.

    audio: raw audio bytes, e.g. Streamlit's `st.audio_input(...).read()`.
        Normal, real-usage path — runs STT first.

    query_text/language: bypass STT with an already-known query. Used for
        latency benchmarking and for text-only debugging/demoing without a
        microphone.
    """
    state = PipelineState(audio=audio)

    # ---- Stage 1: Speech-to-text (Person 1) ----
    if audio is not None:
        t0 = time.time()
        try:
            stt_result = with_retry(transcribe, audio)
        except Exception:
            state.query_ok = False
            state.reject_reason = TRY_AGAIN_MESSAGE
            state.timings["stt_ms"] = (time.time() - t0) * 1000
            return state
        state.timings["stt_ms"] = (time.time() - t0) * 1000
        state.transcript = stt_result["transcript"] or ""
        state.language = stt_result["language"] or ""
    else:
        state.transcript = query_text or ""
        state.language = language or "en"
        state.timings["stt_ms"] = 0.0

    if not state.transcript.strip():
        state.query_ok = False
        state.reject_reason = TRY_AGAIN_MESSAGE
        return state

    # ---- Stage 2: Input guardrails (Person 1) ----
    t1 = time.time()
    try:
        off = with_retry(check_off_topic, state.transcript)
        unsafe = with_retry(check_unsafe, state.transcript)
    except Exception:
        # Fail safe: if the classifier itself is unreachable, don't let an
        # unchecked query through to retrieval/generation.
        state.query_ok = False
        state.reject_reason = TRY_AGAIN_MESSAGE
        state.timings["input_guardrails_ms"] = (time.time() - t1) * 1000
        return state
    state.timings["input_guardrails_ms"] = (time.time() - t1) * 1000

    if unsafe["unsafe"]:
        state.query_ok = False
        state.reject_reason = f"unsafe content ({unsafe['category']})"
        return state
    if not off["in_domain"]:
        state.query_ok = False
        state.reject_reason = off["reason"]
        return state

    # ---- Stage 3: Retrieval (Person 2) ----
    t2 = time.time()
    try:
        state.retrieved = with_retry(engine.retrieve, state.transcript, state.language, top_k=top_k)
    except Exception:
        state.retrieved = []  # empty retrieval -> routed to low-confidence branch below
    state.timings["retrieval_ms"] = (time.time() - t2) * 1000

    top_score = state.retrieved[0]["score"] if state.retrieved else 0.0
    if not state.retrieved or not check_retrieval_confidence(top_score, confidence_threshold):
        state.query_ok = False
        state.reject_reason = "no confident match in knowledge base"
        return state

    # ---- Stage 4: Generation (Person 3) ----
    t3 = time.time()
    try:
        gen = with_retry(generate_answer, state.transcript, state.retrieved, retries=1)
    except Exception:
        state.query_ok = False
        state.reject_reason = "generation failed — please try again"
        state.timings["generation_ms"] = (time.time() - t3) * 1000
        return state
    state.timings["generation_ms"] = (time.time() - t3) * 1000
    state.answer = gen.get("answer", "")
    state.sources = gen.get("sources", [])

    # ---- Stage 5: Output guardrail — grounding (Person 1) ----
    t4 = time.time()
    try:
        check = with_retry(check_grounded, state.transcript, state.retrieved, state.answer, retries=1)
        state.grounded = bool(check.get("grounded", True))
    except Exception:
        # If the grounding checker is unreachable, fail safe rather than
        # show an answer that was never actually verified.
        state.grounded = False
    state.timings["grounding_ms"] = (time.time() - t4) * 1000

    if not state.grounded:
        state.answer = UNGROUNDED_MESSAGE

    return state
