#!/usr/bin/env python3
"""
Quick sanity tests for the retrieval engine.
Run this AFTER build_index.py has successfully created the index.

Usage:
    python test_retrieval.py

Env vars:
    DB_PATH="./vector_index"
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval_engine import RetrievalEngine, DEFAULT_DB_PATH


TEST_QUERIES = [
    # (query_text, expected_language)
    ("What is the capital of France?", "en"),
    ("भारत की राजधानी क्या है?", "hi"),
    ("How does photosynthesis work?", "en"),
    ("Who wrote the Indian constitution?", "en"),
    ("சூரியன் என்றால் என்ன?", "ta"),  # "What is the sun?" (Tamil)
]

OFF_TOPIC_QUERIES = [
    ("What's 348 times 27?", "en"),
    ("Write me a Python script to sort a list.", "en"),
]


def main():
    db_path = os.environ.get("DB_PATH", DEFAULT_DB_PATH)
    print("Loading retrieval engine...")
    engine = RetrievalEngine(db_path=db_path)

    stats = engine.get_stats()
    print(f"Index stats: {stats}")
    print()

    print("=" * 60)
    print("IN-DOMAIN QUERIES")
    print("=" * 60)
    for query, lang in TEST_QUERIES:
        print(f"\n📝 Query: {query!r} (lang={lang})")
        results = engine.retrieve(query, language=lang, top_k=3)
        if not results:
            print("   ⚠️  No results returned.")
            continue
        for i, r in enumerate(results, 1):
            print(f"   #{i} [score={r['score']:.3f} | strategy={r['strategy']}] {r['text'][:120]}...")

    print()
    print("=" * 60)
    print("OFF-TOPIC QUERIES (expect low scores or empty)")
    print("=" * 60)
    for query, lang in OFF_TOPIC_QUERIES:
        print(f"\n📝 Query: {query!r} (lang={lang})")
        results = engine.retrieve(query, language=lang, top_k=3)
        if not results:
            print("   ⚠️  No results returned.")
            continue
        for i, r in enumerate(results, 1):
            print(f"   #{i} [score={r['score']:.3f} | strategy={r['strategy']}] {r['text'][:120]}...")

    print()
    print("=" * 60)
    print("LANGUAGE FILTER TEST")
    print("=" * 60)
    # Query in Hindi but search English index — should return nothing or poor results
    query_hi = "भारत की राजधानी क्या है?"
    print(f"\n📝 Query: {query_hi!r} (lang=en — WRONG language filter)")
    results = engine.retrieve(query_hi, language="en", top_k=3)
    if not results:
        print("   ✅ Correctly returned no results (language filter working).")
    else:
        print(f"   ⚠️  Returned {len(results)} results (unexpected — check index).")
        for r in results:
            print(f"      {r['text'][:100]}...")

    print("\n✅ Test run complete.")


if __name__ == "__main__":
    main()
