#!/usr/bin/env python3
"""
Offline index builder for Person 2.
Run this ONCE before the demo to build the persistent vector index.

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

# Add current dir to path so retrieval_engine imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval_engine import RetrievalEngine, DEFAULT_DB_PATH, DEFAULT_LANGUAGES


def main():
    # Parse env overrides
    languages_env = os.environ.get("LANGUAGES", ",".join(DEFAULT_LANGUAGES))
    languages = [l.strip() for l in languages_env.split(",") if l.strip()]
    max_examples = int(os.environ.get("MAX_EXAMPLES", "1000"))
    db_path = os.environ.get("DB_PATH", DEFAULT_DB_PATH)

    print("=" * 60)
    print("MSMARCO-XI Index Builder")
    print("=" * 60)
    print(f"Languages      : {languages}")
    print(f"Max examples   : {max_examples} per language")
    print(f"DB path        : {db_path}")
    print(f"This will download embedding models on first run.")
    print("=" * 60)
    print()

    engine = RetrievalEngine(db_path=db_path)
    engine.build_index(languages=languages, max_examples_per_lang=max_examples)

    # Print stats
    stats = engine.get_stats()
    print("\n--- Index Stats ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n✅ Done. You can now run test_retrieval.py or wire this into the harness.")


if __name__ == "__main__":
    main()
