#!/usr/bin/env python3
"""
Person 3 — Full end-to-end latency benchmark.

Produces the SECOND of the two README latency tables (the first,
retrieval-only, is benchmark_retrieval.py — run that too and copy both
tables in).

Honesty note (per the playbook: "report honestly... judges respect an
honest instrumented number far more than a suspiciously perfect one"):
recording 30+ real spoken audio clips wasn't practical for an automated
benchmark run, so the "full pipeline" numbers below cover every stage
EXCEPT speech-to-text — input guardrails, retrieval, generation, and the
grounding check — run against the SAME 30 text queries Person 2 already
benchmarked retrieval with (imported from benchmark_retrieval.py, not
redefined, so the two tables are directly comparable).

To get a truly complete spoken-to-answer number: measure one or two real
STT round-trips with `python test_sarvam_stt.py <clip>` and add that
figure on top when writing up results — STT latency is dominated by
network + audio length, not by anything this pipeline controls, so a
couple of real samples is enough to characterize it honestly.

Usage:
    python benchmark_pipeline.py

Env vars:
    DB_PATH="./vector_index"
    N_QUERIES=30
"""

import os
import time
import statistics

from retrieval_engine import RetrievalEngine, DEFAULT_DB_PATH
from benchmark_retrieval import BENCHMARK_QUERIES, percentile
from pipeline import run_pipeline


def main():
    db_path = os.environ.get("DB_PATH", DEFAULT_DB_PATH)
    n_queries = int(os.environ.get("N_QUERIES", str(len(BENCHMARK_QUERIES))))

    print("Loading retrieval engine (embedder + re-ranker + index)...")
    engine = RetrievalEngine(db_path=db_path)

    queries = BENCHMARK_QUERIES[:n_queries]
    print(f"\nRunning FULL PIPELINE benchmark with {len(queries)} queries.")
    print("(STT stage bypassed — text input used directly. See module docstring.)\n")

    total_times = []
    stage_times = {"input_guardrails_ms": [], "retrieval_ms": [], "generation_ms": [], "grounding_ms": []}
    rejected = 0

    for query, lang in queries:
        t0 = time.time()
        state = run_pipeline(engine, query_text=query, language=lang)
        total_ms = (time.time() - t0) * 1000

        total_times.append(total_ms)
        for k in stage_times:
            stage_times[k].append(state.timings.get(k, 0.0))
        if not state.query_ok:
            rejected += 1

    def pct(vals, p):
        return round(percentile(vals, p), 2) if vals else 0.0

    print("=" * 70)
    print("FULL END-TO-END LATENCY RESULTS  (STT excluded — see note above)")
    print("=" * 70)
    print(f"{'Stage':<24} {'P50 (ms)':<12} {'P70 (ms)':<12} {'P100 (ms)':<12}")
    print("-" * 70)
    print(f"{'Full pipeline (no STT)':<24} {pct(total_times, 50):<12} {pct(total_times, 70):<12} {pct(total_times, 100):<12}")
    for k, vals in stage_times.items():
        print(f"{k:<24} {pct(vals, 50):<12} {pct(vals, 70):<12} {pct(vals, 100):<12}")
    print()
    print(f"Queries run                      : {len(total_times)}")
    print(f"Queries rejected by guardrails    : {rejected}")
    print(f"Mean total time                  : {statistics.mean(total_times):.2f} ms")
    print(f"Min total time                   : {min(total_times):.2f} ms")
    print(f"Max total time                   : {max(total_times):.2f} ms")
    print()
    print("Copy this table into the README under 'Full End-to-End Latency'.")
    print("Add a real STT round-trip (test_sarvam_stt.py on an actual clip) on")
    print("top of this if you want a complete spoken-question-to-answer number.")


if __name__ == "__main__":
    main()
