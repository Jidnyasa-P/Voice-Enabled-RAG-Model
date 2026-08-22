"""
Test harness for the OUTPUT guardrails: check_grounded() and
check_retrieval_confidence() / validate_answer().

Unlike check_off_topic/check_unsafe, these need retrieved context +
a generated answer to check against — which Person 2/3 haven't built
yet. So this script uses realistic MOCK data (fake retrieved chunks,
fake answers — one grounded, one hallucinated) to prove the guardrail
logic works standalone. Once Person 2's retrieve() and Person 3's
generate_answer() are real, swap the mock data below for their actual
output — the function signatures don't change.

Usage:
    export SARVAM_API_KEY="your_key"
    python test_grounding.py
"""

from guardrails import check_grounded, check_retrieval_confidence, validate_answer

# ---------------------------------------------------------------------
# Mock "retrieved context" — stands in for Person 2's retrieve() output
# ---------------------------------------------------------------------
MOCK_CONTEXT = [
    {"text": "The Eiffel Tower was completed in 1889 and stands 330 metres tall. "
             "It was designed by engineer Gustave Eiffel for the 1889 World's Fair in Paris."},
    {"text": "The tower is the most-visited paid monument in the world, "
             "attracting nearly 7 million visitors annually."},
]

QUESTION = "When was the Eiffel Tower built and how tall is it?"

# A grounded answer — every claim traces back to the context above
GROUNDED_ANSWER = "The Eiffel Tower was completed in 1889 and is 330 metres tall."

# A hallucinated answer — invents a fact not present in the context
# (the real designer was Gustave Eiffel's company / engineer Maurice Koechlin
# led the design team; "Gustave Eiffel personally welded every beam" is invented)
UNGROUNDED_ANSWER = ("The Eiffel Tower was completed in 1889, stands 330 metres tall, "
                      "and Gustave Eiffel personally welded every beam himself over 3 years.")


def test_grounding():
    print("=== Grounding check ===\n")

    print("Case 1: grounded answer (expect grounded=True)")
    result = check_grounded(QUESTION, MOCK_CONTEXT, GROUNDED_ANSWER)
    status = "✅" if result["grounded"] else "❌"
    print(f"{status} grounded={result['grounded']}  unsupported_claims={result['unsupported_claims']}\n")

    print("Case 2: hallucinated answer (expect grounded=False)")
    result = check_grounded(QUESTION, MOCK_CONTEXT, UNGROUNDED_ANSWER)
    status = "✅" if not result["grounded"] else "❌"
    print(f"{status} grounded={result['grounded']}  unsupported_claims={result['unsupported_claims']}\n")


def test_confidence_threshold():
    print("=== Retrieval confidence threshold ===\n")
    # Simulated similarity scores you'd expect from a real embedding
    # model: high for a good match, low for a query the corpus has
    # nothing relevant for. Once Person 2's retrieval is live, replace
    # these with real top-1 scores from ~10 in-domain and ~10
    # nothing-relevant queries to pick your actual threshold.
    good_match_score = 0.82
    poor_match_score = 0.31
    borderline_score = 0.58

    for label, score in [("good match", good_match_score),
                          ("poor match", poor_match_score),
                          ("borderline", borderline_score)]:
        passes = check_retrieval_confidence(score, threshold=0.6)
        print(f"{label:>12} (score={score}) -> passes threshold: {passes}")


def test_validate_answer_end_to_end():
    print("\n=== validate_answer() end-to-end ===\n")

    # Case A: good retrieval score + grounded answer -> should pass through
    result = validate_answer(QUESTION, MOCK_CONTEXT, GROUNDED_ANSWER, top_score=0.82)
    print("Case A (confident + grounded):", result)

    # Case B: good retrieval score but hallucinated answer -> should be blocked
    result = validate_answer(QUESTION, MOCK_CONTEXT, UNGROUNDED_ANSWER, top_score=0.82)
    print("Case B (confident + ungrounded):", result)

    # Case C: low retrieval score -> should be blocked before even checking grounding
    result = validate_answer(QUESTION, MOCK_CONTEXT, GROUNDED_ANSWER, top_score=0.31)
    print("Case C (low confidence):", result)


if __name__ == "__main__":
    test_grounding()
    test_confidence_threshold()
    test_validate_answer_end_to_end()
