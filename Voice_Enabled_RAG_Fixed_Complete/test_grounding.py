from guardrails import check_grounded

def test_grounded():
    context = [
        {"text": "The capital of France is Paris. It is known for the Eiffel Tower."},
        {"text": "Paris has a population of over 2 million people."},
    ]
    question = "What is the capital of France?"

    answer1 = "The capital of France is Paris."
    result1 = check_grounded(question, context, answer1)
    print(f"Grounded answer: {result1}")

    answer2 = "The capital of France is Berlin."
    result2 = check_grounded(question, context, answer2)
    print(f"Ungrounded answer: {result2}")

    answer3 = "The capital of France is Paris, which has a population of 10 million."
    result3 = check_grounded(question, context, answer3)
    print(f"Partially grounded: {result3}")

if __name__ == "__main__":
    test_grounded()
