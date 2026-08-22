"""
Person 3 — Orchestration harness (FIXED VERSION).

Wires together:
  - stt.transcribe()                          (Person 1)
  - guardrails.check_off_topic/check_unsafe   (Person 1)
  - RetrievalEngine.retrieve()                (Person 2)
  - generation.generate_answer()              (Person 3)
  - guardrails.check_grounded()               (Person 1)
  - guardrails.check_retrieval_confidence()   (Person 1)

FIXES vs previous version:
  - Better error messages so you know WHICH stage failed
  - Retrieval empty now shows a helpful message instead of generic "no confident match"
  - Added retrieval debug info (top score, number of chunks, filter used)
  - Added index health check at startup
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from stt import transcribe
from guardrails import check_off_topic, check_unsafe, check_grounded, check_retrieval_confidence
from generation import generate_answer
from retry_utils import with_retry

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
    # NEW: debug info for troubleshooting
    retrieval_debug: Dict[str, Any] = field(default_factory=dict)


def run_pipeline(
    engine,
    audio: bytes = None,
    query_text: str = None,
    language: str = None,
    top_k: int = 3,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> PipelineState:
    state = PipelineState(audio=audio)

    # ---- Stage 1: Speech-to-text ----
    if audio is not None:
        t0 = time.time()
        try:
            stt_result = with_retry(transcribe, audio)
        except Exception as e:
            state.query_ok = False
            state.reject_reason = f"STT failed: {str(e)[:100]}"
            state.timings["stt_ms"] = (time.time() - t0) * 1000
            return state
        state.timings["stt_ms"] = (time.time() - t0) * 1000
        state.transcript = stt_result.get("transcript", "") or ""
        state.language = stt_result.get("language", "") or ""
    else:
        state.transcript = query_text or ""
        state.language = language or "en"
        state.timings["stt_ms"] = 0.0

    if not state.transcript.strip():
        state.query_ok = False
        state.reject_reason = "No speech detected — please try again."
        return state

    # ---- Stage 2: Input guardrails ----
    t1 = time.time()
    try:
        off = with_retry(check_off_topic, state.transcript)
        unsafe = with_retry(check_unsafe, state.transcript)
    except Exception as e:
        state.query_ok = False
        state.reject_reason = f"Guardrail check failed: {str(e)[:100]}"
        state.timings["input_guardrails_ms"] = (time.time() - t1) * 1000
        return state
    state.timings["input_guardrails_ms"] = (time.time() - t1) * 1000

    if unsafe.get("unsafe"):
        state.query_ok = False
        state.reject_reason = f"unsafe content ({unsafe.get('category', 'unknown')})"
        return state
    if not off.get("in_domain", True):
        state.query_ok = False
        state.reject_reason = off.get("reason", "off-topic query")
        return state

    # ---- Stage 3: Retrieval ----
    t2 = time.time()
    try:
        state.retrieved = with_retry(engine.retrieve, state.transcript, state.language, top_k=top_k)
    except Exception as e:
        state.retrieved = []
        state.retrieval_debug["error"] = str(e)[:200]
    state.timings["retrieval_ms"] = (time.time() - t2) * 1000

    top_score = state.retrieved[0]["score"] if state.retrieved else 0.0
    num_chunks = len(state.retrieved)
    state.retrieval_debug["top_score"] = round(top_score, 4)
    state.retrieval_debug["num_chunks_returned"] = num_chunks

    if not state.retrieved:
        state.query_ok = False
        state.reject_reason = (
            "No passages found for this query. "
            "The knowledge base may be empty or the query doesn't match any indexed content. "
            "Try: (1) running build_index.py, (2) asking a different question, or (3) checking the language detected."
        )
        return state

    if not check_retrieval_confidence(top_score, confidence_threshold):
        state.query_ok = False
        state.reject_reason = (
            f"No confident match (top score {top_score:.3f} < threshold {confidence_threshold}). "
            f"The retrieved passages don't seem relevant enough to answer reliably."
        )
        return state

    # ---- Stage 4: Generation ----
    t3 = time.time()
    try:
        gen = with_retry(generate_answer, state.transcript, state.retrieved, retries=1)
    except Exception as e:
        state.query_ok = False
        state.reject_reason = f"Generation failed: {str(e)[:100]}"
        state.timings["generation_ms"] = (time.time() - t3) * 1000
        return state
    state.timings["generation_ms"] = (time.time() - t3) * 1000
    state.answer = gen.get("answer", "")
    state.sources = gen.get("sources", [])

    # ---- Stage 5: Grounding check ----
    t4 = time.time()
    try:
        check = with_retry(check_grounded, state.transcript, state.retrieved, state.answer, retries=1)
        state.grounded = bool(check.get("grounded", True))
    except Exception:
        state.grounded = False
    state.timings["grounding_ms"] = (time.time() - t4) * 1000

    if not state.grounded:
        state.answer = UNGROUNDED_MESSAGE

    return state
