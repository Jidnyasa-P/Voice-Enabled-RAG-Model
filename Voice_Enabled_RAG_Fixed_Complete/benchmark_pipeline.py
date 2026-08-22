#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval_engine import RetrievalEngine
from pipeline import run_pipeline

QUERIES = [
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

def percentile(data, pct):
    data = sorted(data)
    k = (len(data) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(data) - 1)
    return data[f] + (data[c] - data[f]) * (k - f)

def main():
    engine = RetrievalEngine()
    retrieval_times, total_times = [], []

    for query, lang in QUERIES:
        state = run_pipeline(engine, query_text=query, language=lang)
        retrieval_times.append(state.timings.get("retrieval_ms", 0))
        total_times.append(sum(state.timings.values()))

    print("=" * 60)
    print("PIPELINE LATENCY RESULTS")
    print("=" * 60)
    for label, data in [("Retrieval-only", retrieval_times), ("Full end-to-end", total_times)]:
        print(f"{label}: P50={percentile(data,50):.1f}ms | P70={percentile(data,70):.1f}ms | P100={percentile(data,100):.1f}ms")

if __name__ == "__main__":
    main()
