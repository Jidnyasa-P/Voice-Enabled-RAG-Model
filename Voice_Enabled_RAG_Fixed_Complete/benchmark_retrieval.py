#!/usr/bin/env python3
import os, sys, time, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval_engine import RetrievalEngine, DEFAULT_DB_PATH

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
    ("What is 348 times 27?", "en"),
    ("Write me a Python script.", "en"),
    ("Tell me a joke.", "en"),
    ("What do you think about me?", "en"),
    ("How do I hack a website?", "en"),
    ("What is the meaning of life?", "en"),
    ("Explain string theory in one sentence.", "en"),
    ("Who won the 2050 World Cup?", "en"),
]

def percentile(values, p):
    if not values: return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c: return s[f]
    return s[f] * (c - k) + s[c] * (k - f)

def main():
    db_path = os.environ.get("DB_PATH", DEFAULT_DB_PATH)
    engine = RetrievalEngine(db_path=db_path)
    queries = BENCHMARK_QUERIES[:int(os.environ.get("N_QUERIES", len(BENCHMARK_QUERIES)))]

    total_times = []
    top_scores = []

    for query, lang in queries:
        t0 = time.time()
        results = engine.retrieve(query, language=lang, top_k=3)
        total_ms = (time.time() - t0) * 1000
        total_times.append(total_ms)
        top_scores.append(results[0]["score"] if results else 0.0)

    def pct(vals, p): return round(percentile(vals, p), 2)

    print("=" * 60)
    print("RETRIEVAL-ONLY LATENCY RESULTS")
    print("=" * 60)
    print(f"{'Metric':<20} {'P50 (ms)':<12} {'P70 (ms)':<12} {'P100 (ms)':<12}")
    print("-" * 60)
    print(f"{'Total retrieval':<20} {pct(total_times, 50):<12} {pct(total_times, 70):<12} {pct(total_times, 100):<12}")
    print(f"\nQueries run       : {len(total_times)}")
    print(f"Mean total time   : {statistics.mean(total_times):.2f} ms")
    print(f"Min total time    : {min(total_times):.2f} ms")
    print(f"Max total time    : {max(total_times):.2f} ms")
    print(f"\nTop score P50     : {pct(top_scores, 50):.3f}")
    print(f"Top score P70     : {pct(top_scores, 70):.3f}")
    print(f"Top score P100    : {pct(top_scores, 100):.3f}")

    p100 = pct(total_times, 100)
    status = "✅ UNDER 200ms" if p100 < 200 else "⚠️  OVER 200ms"
    print(f"\nP100 retrieval    : {p100:.2f} ms  {status}")
    print("\nCopy the table above into your README under 'Latency Benchmarking'.")

if __name__ == "__main__":
    main()
