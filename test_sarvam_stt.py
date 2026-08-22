"""
Quick standalone test for Sarvam Speech-to-Text.
Run this FIRST, before wiring it into anything else, to confirm your key
and setup actually work.

The actual transcribe() implementation lives in stt.py (shared with the
harness in pipeline.py) — this script is just a thin CLI wrapper around it.

Setup:
    pip install sarvamai

Usage:
    1. Record a short (<30s) voice clip on your phone, any format
       (WAV/MP3/M4A/OGG/etc — Sarvam auto-detects most codecs).
    2. AirDrop/transfer it into this folder, e.g. test_clip.m4a
    3. Set your API key: export SARVAM_API_KEY="your_key"
    4. python test_sarvam_stt.py test_clip.m4a
"""

import sys

from stt import transcribe, API_KEY


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_sarvam_stt.py <path_to_audio_file>")
        sys.exit(1)

    if API_KEY == "PASTE_YOUR_API_KEY_HERE":
        print("⚠️  Set your API key: `export SARVAM_API_KEY=...`")
        sys.exit(1)

    audio_file = sys.argv[1]
    out = transcribe(audio_file)

    print("\n--- Result ---")
    print("Transcript:", out["transcript"])
    print("Language:  ", out["language"], " (normalized short code)")
    print("Confidence:", out["confidence"])
