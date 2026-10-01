from sentence_transformers import SentenceTransformer, util


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def normalize_answer(text):
    text = text.strip().lower()

    punctuation = ".,!?;:\"'"

    for character in punctuation:
        text = text.replace(character, "")

    return " ".join(text.split())


def exact_match(expected, actual):
    return normalize_answer(expected) == normalize_answer(actual)


def semantic_similarity(expected, actual):
    expected_embedding = embedding_model.encode(
        expected,
        convert_to_tensor=True
    )

    actual_embedding = embedding_model.encode(
        actual,
        convert_to_tensor=True
    )

    similarity = util.cos_sim(
        expected_embedding,
        actual_embedding
    )

    return similarity.item()