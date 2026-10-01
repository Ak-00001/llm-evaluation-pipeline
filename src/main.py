import logging
from langfuse import get_client
from evaluation import exact_match, semantic_similarity
import os
import csv
import requests
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
langfuse = get_client()

api_key = os.getenv("OLLAMA_API_KEY")

OLLAMA_URL = "https://ollama.com/api/chat"
MODEL = "gpt-oss:20b"
PROMPT_A = """
Answer the user's question accurately and concisely.
"""

PROMPT_B = """
You are a precise question-answering assistant.

Answer the user's question directly.
Give the answer first, followed by a short explanation when useful.
Do not include unnecessary information or alternative interpretations.
"""
PROMPTS = {
    "prompt_a": PROMPT_A,
    "prompt_b": PROMPT_B
}



def ask_llm(question, system_prompt):

    logger.info(f"Sending question: {question}")

    generation = langfuse.start_observation(
        name="ollama-generation",
        as_type="generation",
        input={
            "question": question,
            "system_prompt": system_prompt
        },
        model=MODEL
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            headers={
                "Authorization": f"Bearer {api_key}"
            },
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                "stream": False
            },
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        answer = data["message"]["content"].strip()

        generation.update(output=answer)
        generation.end()

        logger.info("LLM response received successfully")

        return answer

    except requests.exceptions.Timeout:
        logger.error("LLM request timed out")
        generation.update(output={"error": "Request timed out"})
        generation.end()
        raise

    except requests.exceptions.RequestException as error:
        logger.error(f"LLM request failed: {error}")
        generation.update(output={"error": str(error)})
        generation.end()
        raise

    except (KeyError, TypeError, ValueError) as error:
        logger.error(f"Invalid LLM response: {error}")
        generation.update(output={"error": str(error)})
        generation.end()
        raise



def llm_judge(question, expected, actual):
    prompt = f"""
You are an evaluator for an LLM evaluation system.

Your job is to determine whether the ACTUAL ANSWER should be accepted
against the EXPECTED ANSWER.

Question:
{question}

EXPECTED ANSWER (reference):
{expected}

ACTUAL ANSWER (model output):
{actual}

Evaluation rules:

1. The actual answer must correctly answer the question.
2. The actual answer must be consistent with the expected answer.
3. Different wording is allowed.
4. Extra explanation is allowed if it does not introduce incorrect information.
5. If the actual answer contradicts the expected answer, return FAIL.
6. Do not replace the expected answer with your own interpretation of the question.

Respond with exactly one word:
PASS
or
FAIL
"""

    response = requests.post(
        OLLAMA_URL,
        headers={
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        },
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"].strip()

def main():
    all_results = []

    for prompt_name, system_prompt in PROMPTS.items():

        print("\n" + "=" * 60)
        print(f"RUNNING: {prompt_name}")
        print("=" * 60)

        correct = 0
        total = 0
        semantic_total = 0
        judge_passes = 0

        with open("data/golden_dataset.csv", "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                question = row["question"]
                expected_answer = row["expected_answer"]

                actual_answer = ask_llm(question, system_prompt)

                is_correct = exact_match(
                    expected_answer,
                    actual_answer
                )

                similarity_score = semantic_similarity(
                    expected_answer,
                    actual_answer
                )

                semantic_total += similarity_score

                judge_result = llm_judge(
                    question,
                    expected_answer,
                    actual_answer
                )

                if judge_result.strip().upper() == "PASS":
                    judge_passes += 1

                total += 1

                if is_correct:
                    correct += 1

                print("-" * 60)
                print(f"ID: {row['id']}")
                print(f"Question: {question}")
                print(f"Expected: {expected_answer}")
                print(f"Actual: {actual_answer}")
                print(f"Exact match: {is_correct}")
                print(f"Semantic similarity: {similarity_score:.3f}")
                print(f"LLM judge: {judge_result}")

                all_results.append({
                    "prompt": prompt_name,
                    "id": row["id"],
                    "question": question,
                    "expected_answer": expected_answer,
                    "actual_answer": actual_answer,
                    "exact_match": is_correct,
                    "semantic_similarity": similarity_score,
                    "llm_judge": judge_result
                })

        exact_accuracy = correct / total * 100
        semantic_average = semantic_total / total
        judge_accuracy = judge_passes / total * 100

        print("\n" + "-" * 60)
        print(f"{prompt_name} SUMMARY")
        print("-" * 60)
        print(
            f"Exact-match accuracy: "
            f"{correct}/{total} = {exact_accuracy:.1f}%"
        )
        print(
            f"Average semantic similarity: "
            f"{semantic_average:.3f}"
        )
        print(
            f"LLM-judge accuracy: "
            f"{judge_passes}/{total} = {judge_accuracy:.1f}%"
        )

    with open(
        "data/results.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "prompt",
            "id",
            "question",
            "expected_answer",
            "actual_answer",
            "exact_match",
            "semantic_similarity",
            "llm_judge"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\nAll results saved to data/results.csv")

if __name__ == "__main__":
    main()