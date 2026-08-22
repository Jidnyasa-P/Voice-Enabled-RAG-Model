#!/usr/bin/env python3
"""
Debug / demo panel showing which chunking strategy retrieved the winning passage.

This is the "visible chunking" feature from the roadmap:
  "add a small debug panel showing which chunking strategy retrieved
   the winning passage, so the 'vast chunking' claim is demonstrable live."

Usage:
    python debug_panel.py "your query here" --language hi

Output:
    Pretty-printed table showing:
    - Query and detected language
    - Top-3 retrieved chunks with their strategy, score, and text preview
    - Which strategy "won" (delivered the top result)
    - Strategy distribution across top-3
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval_engine import RetrievalEngine, DEFAULT_DB_PATH


def print_box(title, content_lines):
    """Print a bordered box for terminal display."""
    width = max(len(title), max((len(line) for line in content_lines), default=0)) + 4
    print("┌" + "─" * width + "┐")
    print("│ " + title.center(width - 2) + " │")
    print("├" + "─" * width + "┤")
    for line in content_lines:
        print("│ " + line.ljust(width - 2) + " │")
    print("└" + "─" * width + "┘")


def main():
    parser = argparse.ArgumentParser(description="Debug panel for chunking strategy visibility")
    parser.add_argument("query", type=str, help="Query text to search")
    parser.add_argument("--language", "-l", type=str, default="en", help="Language code (e.g. hi, en, ta)")
    parser.add_argument("--top_k", "-k", type=int, default=3, help="Number of results to show")
    parser.add_argument("--db_path", "-d", type=str, default=DEFAULT_DB_PATH, help="Chroma DB path")
    args = parser.parse_args()

    print("Loading retrieval engine...")
    engine = RetrievalEngine(db_path=args.db_path)

    print(f"\n🔍 Query: {args.query!r}")
    print(f"🌐 Language filter: {args.language}")
    print("-" * 60)

    results = engine.retrieve(args.query, language=args.language, top_k=args.top_k)

    if not results:
        print("\n⚠️  No results found.")
        return

    # Strategy distribution
    strategies = {}
    for r in results:
        s = r["strategy"]
        strategies[s] = strategies.get(s, 0) + 1

    # Winner
    winner_strategy = results[0]["strategy"]

    lines = []
    lines.append(f"Winner strategy: {winner_strategy.upper()}")
    lines.append("")
    for i, r in enumerate(results, 1):
        badge = "🏆" if i == 1 else f"  #{i}"
        text_preview = r["text"][:80].replace("\n", " ")
        lines.append(f"{badge} [{r['strategy']:<10}] score={r['score']:.3f} | {text_preview}...")
    lines.append("")
    lines.append("Strategy distribution in top-3:")
    for s, count in sorted(strategies.items(), key=lambda x: -x[1]):
        bar = "█" * count
        lines.append(f"  {s:<12} {bar} ({count})")

    print_box("RETRIEVAL DEBUG PANEL", lines)

    print("\n💡 Tip: Show this panel during your demo to prove multi-strategy chunking is real.")


if __name__ == "__main__":
    main()
