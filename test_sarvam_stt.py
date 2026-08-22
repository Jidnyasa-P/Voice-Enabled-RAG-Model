"""
Quick standalone test for Sarvam Speech-to-Text.
Run this FIRST, before wiring it into anything else, to confirm your key
and setup actually work.

Setup:
    pip install sarvamai

Usage:
    1. Record a short (<30s) voice clip on your phone, any format
       (WAV/MP3/M4A/OGG/etc — Sarvam auto-detects most codecs).
    2. AirDrop/transfer it into this folder, e.g. test_clip.m4a
    3. Set your API key below (or export SARVAM_API_KEY as an env var).
    4. python test_sarvam_stt.py test_clip.m4a
"""

import sys
import os
from sarvamai import SarvamAI

API_KEY = os.environ.get("SARVAM_API_KEY", "PASTE_YOUR_API_KEY_HERE")


def transcribe(audio_path: str, language_code: str = "unknown") -> dict:
    """
    Contract for the rest of the team:
        transcribe(audio_path) -> {transcript, language, confidence}

    language_code="unknown" lets Sarvam auto-detect the spoken language —
    good default for a demo where different team members speak different
    languages. Pass an explicit code (e.g. "hi-IN") if you already know it.
    """
    client = SarvamAI(api_subscription_key=API_KEY)

    with open(audio_path, "rb") as f:
        result = client.speech_to_text.transcribe(
            file=f,
            model="saaras:v4",       # latest model as of Aug 2026; saarika:v2.5 is legacy/deprecating
            language_code=language_code,
            mode="transcribe",       # vs "translate" (-> English), "codemix", "translit", "verbatim"
        )

    return {
        "transcript": result.transcript,
        "language": result.language_code,
        # only present when language_code="unknown" / omitted
        "confidence": getattr(result, "language_probability", None),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_sarvam_stt.py <path_to_audio_file>")
        sys.exit(1)

    if API_KEY == "PASTE_YOUR_API_KEY_HERE":
        print("⚠️  Set your API key: edit API_KEY above, or `export SARVAM_API_KEY=...`")
        sys.exit(1)

    audio_file = sys.argv[1]
    out = transcribe(audio_file)

    print("\n--- Result ---")
    print("Transcript:", out["transcript"])
    print("Language:  ", out["language"])
    print("Confidence:", out["confidence"])
