#!/usr/bin/env python3
"""
Offline index builder for Person 2 (FIXED VERSION).
Run this ONCE before the demo to build the persistent vector index.

CRITICAL: If you skip this step, retrieval will return empty results
and the app will show "no confident match in knowledge base" for
EVERY query.

Usage:
    python build_index.py

Optional env vars:
    LANGUAGES="hi,en,ta"          # comma-separated language codes
    MAX_EXAMPLES=1000              # per language
    DB_PATH="./vector_index"       # where Chroma persists

Example:
    LANGUAGES="hi,en" MAX_EXAMPLES=500 python build_index.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval_engine import RetrievalEngine, DEFAULT_DB_PATH, DEFAULT_LANGUAGES


def main():
    languages_env = os.environ.get("LANGUAGES", ",".join(DEFAULT_LANGUAGES))
    languages = [l.strip() for l in languages_env.split(",") if l.strip()]
    max_examples = int(os.environ.get("MAX_EXAMPLES", "1000"))
    db_path = os.environ.get("DB_PATH", DEFAULT_DB_PATH)

    print("=" * 70)
    print("MSMARCO-XI Index Builder")
    print("=" * 70)
    print(f"Languages      : {languages}")
    print(f"Max examples   : {max_examples} per language")
    print(f"DB path        : {db_path}")
    print(f"\n⚠️  This will download embedding models on first run (~1.2 GB).")
    print(f"⚠️  First run takes 10–30 minutes depending on internet & CPU.")
    print(f"⚠️  DO NOT interrupt this script. Wait for the 'COMPLETE' message.")
    print("=" * 70)
    print()

    # Check if index already exists
    if os.path.exists(db_path):
        print(f"📁 DB path already exists: {db_path}")
        print("   If you want to rebuild from scratch, delete this folder first.")
        print("   Continuing will add to the existing index (not replace it).\n")

    engine = RetrievalEngine(db_path=db_path)

    # Pre-flight health check
    health = engine.index_health_check()
    print(f"Pre-build health check: {health['status']}")
    print(f"Current chunks: {health['total_chunks']}")
    print()

    engine.build_index(languages=languages, max_examples_per_lang=max_examples)

    # Post-build validation
    print("\n" + "=" * 70)
    print("POST-BUILD VALIDATION")
    print("=" * 70)
    health = engine.index_health_check()
    print(f"Status : {health['status']}")
    print(f"Chunks : {health['total_chunks']}")
    print(f"Langs  : {health['languages']}")
    print(f"Strats : {health['sample_strategies']}")

    if health['total_chunks'] == 0:
        print("\n❌❌❌ BUILD FAILED — INDEX IS EMPTY ❌❌❌")
        print("Check the error messages above.")
        sys.exit(1)

    # Run a quick sanity retrieval
    print("\n--- Running sanity retrieval test ---")
    engine.quick_test("What is the capital of India?", language="en")

    print("\n✅✅✅ BUILD COMPLETE ✅✅✅")
    print("You can now run: python test_retrieval.py")
    print("Or start the app: streamlit run app.py")


if __name__ == "__main__":
    main()
