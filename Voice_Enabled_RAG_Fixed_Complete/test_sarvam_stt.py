import sys, os
from stt import transcribe

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_sarvam_stt.py <path_to_audio_file>")
        sys.exit(1)
    audio_file = sys.argv[1]
    out = transcribe(audio_file)
    print("\n--- Result ---")
    print("Transcript:", out["transcript"])
    print("Language:  ", out["language"])
    print("Confidence:", out["confidence"])
