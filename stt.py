"""
Person 1 — Speech-to-Text (Sarvam), shared module.

This is the transcribe() function refactored out of test_sarvam_stt.py so
Person 3's harness (pipeline.py) can import it directly, instead of the
harness depending on a script that was only meant to be run standalone.
test_sarvam_stt.py now imports transcribe() from here instead of defining
its own copy — one implementation, not two that can drift apart.

Two things were adjusted vs. the original test-script version, to match
how the rest of the pipeline actually calls this:

1. Accepts raw audio BYTES, not just a file path. Streamlit's
   `st.audio_input()` hands the harness bytes (`audio.read()`), not a
   path on disk — the original transcribe(audio_path: str) would have
   raised inside app.py. This version accepts a path, raw bytes, or an
   already-open file-like object.

2. Normalizes the returned language code. Sarvam returns BCP-47-ish
   codes like "hi-IN" / "en-IN" / "ta-IN" (see the original docstring:
   "e.g. hi-IN"). retrieval_engine.py's index and its `retrieve()`
   language filter are keyed on short codes ("hi", "en", "ta") to match
   MSMARCO-XI's language identifiers. Without normalizing here, the
   detected language would never match anything in the vector index and
   every query would silently fall through the confidence guardrail.
   Normalizing once, at the source, keeps STT output and retrieval's
   language filter speaking the same format everywhere downstream.

Contract:
    transcribe(audio) -> {"transcript": str, "language": str, "confidence": float | None}
"""

import io
import os
from sarvamai import SarvamAI

API_KEY = os.environ.get("SARVAM_API_KEY", "PASTE_API_KEY_HERE")

_client = None


def _get_client() -> SarvamAI:
    global _client
    if _client is None:
        _client = SarvamAI(api_subscription_key=API_KEY)
    return _client


def _normalize_language_code(code: str) -> str:
    """'hi-IN' / 'hin_Deva' / 'HI' -> 'hi' — match retrieval's short codes."""
    if not code:
        return ""
    return code.split("-")[0].split("_")[0].lower()


def transcribe(audio, language_code: str = "unknown") -> dict:
    """
    Contract for the rest of the team:
        transcribe(audio) -> {transcript: str, language: str, confidence: float | None}

    `audio` may be:
      - a file path (str)
      - raw audio bytes (bytes/bytearray) — e.g. from st.audio_input().read()
      - an already-open file-like object

    language_code="unknown" lets Sarvam auto-detect the spoken language —
    good default for a demo where different team members speak different
    languages. Pass an explicit code (e.g. "hi-IN") if you already know it.
    """
    client = _get_client()

    close_after = False
    if isinstance(audio, (bytes, bytearray)):
        f = io.BytesIO(audio)
        f.name = "audio.wav"  # Sarvam sniffs the codec from the filename/extension
    elif isinstance(audio, str):
        f = open(audio, "rb")
        close_after = True
    else:
        f = audio  # assume already a file-like object

    try:
        result = client.speech_to_text.transcribe(
            file=f,
            model="saaras:v4",       # latest model as of Aug 2026; saarika:v2.5 is legacy/deprecating
            language_code=language_code,
            mode="transcribe",       # vs "translate" (-> English), "codemix", "translit", "verbatim"
        )
    finally:
        if close_after:
            f.close()

    return {
        "transcript": result.transcript,
        "language": _normalize_language_code(result.language_code),
        # only meaningfully populated when language_code="unknown" / omitted
        "confidence": getattr(result, "language_probability", None),
    }
