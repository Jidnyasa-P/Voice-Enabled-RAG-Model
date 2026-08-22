#!/usr/bin/env python3
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval_engine import RetrievalEngine, DEFAULT_DB_PATH

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--language", "-l", type=str, default="en")
    parser.add_argument("--top_k", "-k", type=int, default=3)
    parser.add_argument("--db_path", "-d", type=str, default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    engine = RetrievalEngine(db_path=args.db_path)
    print(f"\n🔍 Query: {args.query!r} (lang={args.language})")
    results = engine.retrieve(args.query, language=args.language, top_k=args.top_k)

    if not results:
        print("⚠️  No results.")
        return

    winner = results[0]["strategy"]
    print(f"🏆 Winner strategy: {winner.upper()}")
    for i, r in enumerate(results, 1):
        badge = "🏆" if i == 1 else f"  #{i}"
        print(f"{badge} [{r['strategy']:<10}] score={r['score']:.3f} | {r['text'][:120]}...")

if __name__ == "__main__":
    main()
