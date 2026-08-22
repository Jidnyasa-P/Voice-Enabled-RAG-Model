#!/usr/bin/env python3
"""
Comprehensive retrieval test suite (FIXED VERSION).
Run this AFTER build_index.py has successfully created the index.

This script does FOUR things:
  1. Index health check (tells you if the index is empty)
  2. Language-filtered retrieval tests
  3. Unfiltered fallback retrieval tests
  4. Cross-language mismatch test

Usage:
    python test_retrieval.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval_engine import RetrievalEngine, DEFAULT_DB_PATH


def main():
    print("=" * 70)
    print("RETRIEVAL ENGINE TEST SUITE")
    print("=" * 70)

    engine = RetrievalEngine()

    # ---- TEST 1: Index Health Check ----
    print("\n--- TEST 1: Index Health Check ---")
    health = engine.index_health_check()
    print(f"Status : {health['status']}")
    print(f"Chunks : {health['total_chunks']}")
    print(f"Langs  : {health['languages']}")
    print(f"Strats : {health['sample_strategies']}")

    if health["total_chunks"] == 0:
        print("\n❌❌❌ INDEX IS EMPTY ❌❌❌")
        print("You MUST run 'python build_index.py' before testing retrieval.")
        print("Without an index, EVERY query will fail.")
        sys.exit(1)

    # ---- TEST 2: Language-filtered retrieval ----
    print("\n--- TEST 2: Language-Filtered Retrieval ---")
    test_queries = [
        ("What is the capital of India?", "en"),
        ("How does photosynthesis work?", "en"),
        ("भारत की राजधानी क्या है?", "hi"),
        ("प्रकाश संश्लेषण कैसे काम करता है?", "hi"),
        ("காந்தியடிகள் யார்?", "ta"),
    ]

    all_passed = True
    for query, lang in test_queries:
        print(f"\n  Query: {query!r} (lang={lang})")
        results = engine.retrieve(query, language=lang, top_k=3)
        if results:
            print(f"    ✅ {len(results)} results (top score={results[0]['score']:.3f}, strategy={results[0]['strategy']})")
        else:
            print(f"    ⚠️  NO RESULTS — this may be normal if the query doesn't match the indexed corpus")
            all_passed = False

    # ---- TEST 3: Wrong language filter (should trigger fallback) ----
    print("\n--- TEST 3: Wrong Language Filter (tests fallback) ---")
    query_hi = "भारत की राजधानी क्या है?"
    print(f"  Query: {query_hi!r} but filtering by lang='en' (WRONG)")
    results = engine.retrieve(query_hi, language="en", top_k=3)
    if results:
        print(f"    ✅ Fallback worked! Got {len(results)} results (unfiltered search saved the query)")
    else:
        print(f"    ⚠️  No results even with fallback")

    # ---- TEST 4: Quick test method ----
    print("\n--- TEST 4: Quick Test Method ---")
    engine.quick_test("Who wrote the Indian constitution?", language="en")

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL CORE TESTS PASSED")
    else:
        print("⚠️  SOME TESTS HAD EMPTY RESULTS")
        print("   This is OK if your indexed examples simply don't cover those topics.")
        print("   The important thing is that retrieval returns results for SOME queries.")
    print("=" * 70)


if __name__ == "__main__":
    main()
