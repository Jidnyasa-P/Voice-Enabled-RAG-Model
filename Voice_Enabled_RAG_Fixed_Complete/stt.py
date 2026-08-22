"""
Person 1 — Speech-to-Text (Sarvam), shared module.

Contract:
    transcribe(audio) -> {transcript: str, language: str, confidence: float | None}

`audio` may be:
  - a file path (str)
  - raw audio bytes (bytes/bytearray)
  - an already-open file-like object

language_code="unknown" lets Sarvam auto-detect the spoken language.
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
    client = _get_client()

    close_after = False
    if isinstance(audio, (bytes, bytearray)):
        f = io.BytesIO(audio)
        f.name = "audio.wav"
    elif isinstance(audio, str):
        f = open(audio, "rb")
        close_after = True
    else:
        f = audio

    try:
        result = client.speech_to_text.transcribe(
            file=f,
            model="saaras:v4",
            language_code=language_code,
            mode="transcribe",
        )
    finally:
        if close_after:
            f.close()

    return {
        "transcript": result.transcript,
        "language": _normalize_language_code(result.language_code),
        "confidence": getattr(result, "language_probability", None),
    }
