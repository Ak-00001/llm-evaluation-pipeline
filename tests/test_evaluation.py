from src.evaluation import (
    normalize_answer,
    exact_match,
    semantic_similarity
)

def test_normalize_answer():
    assert normalize_answer(" Berlin. ") == "berlin"


def test_exact_match():
    assert exact_match("Berlin", "Berlin.") is True


def test_exact_match_wrong_answer():
    assert exact_match("Berlin", "Munich") is False


def test_semantic_similarity():
    score = semantic_similarity(
        "The capital of Germany is Berlin.",
        "Berlin is the capital of Germany."
    )

    assert score > 0.7