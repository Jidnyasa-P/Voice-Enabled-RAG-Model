#!/usr/bin/env python3
"""
Latency benchmark for retrieval-only (Person 2 leg).
Run this AFTER the index is built.

Produces P50 / P70 / P100 tables for the README.
Person 3 can extend this to benchmark the full pipeline.

Usage:
    python benchmark_retrieval.py

Env vars:
    DB_PATH="./vector_index"
    N_QUERIES=30
"""

import os
import sys
import time
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval_engine import RetrievalEngine, DEFAULT_DB_PATH

# Mix of easy / hard / off-topic / unanswerable queries
BENCHMARK_QUERIES = [
    ("What is the capital of France?", "en"),
    ("How does photosynthesis work?", "en"),
    ("Who wrote the Indian constitution?", "en"),
    ("What is quantum mechanics?", "en"),
    ("Explain gravity.", "en"),
    ("What causes earthquakes?", "en"),
    ("How do vaccines work?", "en"),
    ("What is the tallest mountain?", "en"),
    ("Who invented the telephone?", "en"),
    ("What is DNA?", "en"),
    ("How does a car engine work?", "en"),
    ("What is climate change?", "en"),
    ("Who was Mahatma Gandhi?", "en"),
    ("What is the speed of light?", "en"),
    ("How do computers work?", "en"),
    ("भारत की राजधानी क्या है?", "hi"),
    ("प्रकाश संश्लेषण कैसे काम करता है?", "hi"),
    ("भारतीय संविधान किसने लिखा?", "hi"),
    ("भूकंप क्यों आते हैं?", "hi"),
    ("महात्मा गांधी कौन थे?", "hi"),
    ("காந்தியடிகள் யார்?", "ta"),
    ("சூரியன் என்றால் என்ன?", "ta"),
    ("What is 348 times 27?", "en"),          # off-topic / unanswerable from corpus
    ("Write me a Python script.", "en"),      # off-topic
    ("Tell me a joke.", "en"),                # off-topic
    ("What do you think about me?", "en"),    # off-topic
    ("How do I hack a website?", "en"),        # unsafe (retrieval may still run)
    ("What is the meaning of life?", "en"),   # philosophical / hard
    ("Explain string theory in one sentence.", "en"),
    ("Who won the 2050 World Cup?", "en"),    # unanswerable (future)
]


def percentile(values, p):
    """Compute percentile (0-100)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_vals) else f
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def main():
    db_path = os.environ.get("DB_PATH", DEFAULT_DB_PATH)
    n_queries = int(os.environ.get("N_QUERIES", str(len(BENCHMARK_QUERIES))))

    print("Loading retrieval engine...")
    engine = RetrievalEngine(db_path=db_path)

    queries = BENCHMARK_QUERIES[:n_queries]
    print(f"Running benchmark with {len(queries)} queries...\n")

    embed_times = []
    vector_times = []
    rerank_times = []
    total_times = []
    top_scores = []

    for query, lang in queries:
        t0 = time.time()
        results = engine.retrieve(query, language=lang, top_k=3)
        total_ms = (time.time() - t0) * 1000

        # We need to re-run to get per-stage timing without modifying retrieve()
        # Actually, retrieve() already prints per-stage times. Let's capture them
        # by re-running and parsing — simpler: just use total for now.
        # For more accurate per-stage, we can call the internal steps directly.

        total_times.append(total_ms)
        if results:
            top_scores.append(results[0]["score"])
        else:
            top_scores.append(0.0)

    # Compute percentiles
    def pct(vals, p):
        return round(percentile(vals, p), 2)

    print("=" * 60)
    print("RETRIEVAL-ONLY LATENCY RESULTS")
    print("=" * 60)
    print(f"{'Metric':<20} {'P50 (ms)':<12} {'P70 (ms)':<12} {'P100 (ms)':<12}")
    print("-" * 60)
    print(f"{'Total retrieval':<20} {pct(total_times, 50):<12} {pct(total_times, 70):<12} {pct(total_times, 100):<12}")
    print()
    print(f"Queries run       : {len(total_times)}")
    print(f"Mean total time   : {statistics.mean(total_times):.2f} ms")
    print(f"Min total time    : {min(total_times):.2f} ms")
    print(f"Max total time    : {max(total_times):.2f} ms")
    print()
    print(f"Top score P50     : {pct(top_scores, 50):.3f}")
    print(f"Top score P70     : {pct(top_scores, 70):.3f}")
    print(f"Top score P100    : {pct(top_scores, 100):.3f}")
    print()

    # Judge-facing summary
    p100 = pct(total_times, 100)
    status = "✅ UNDER 200ms" if p100 < 200 else "⚠️  OVER 200ms — consider reducing top_k or dropping a strategy"
    print(f"P100 retrieval    : {p100:.2f} ms  {status}")
    print()
    print("Copy the table above into your README under 'Latency Benchmarking'.")


if __name__ == "__main__":
    main()
