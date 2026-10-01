import pandas as pd


df = pd.read_csv("data/results.csv")

print("\nFULL RESULTS")
print(df)

print("\nPROMPT COMPARISON")

summary = df.groupby("prompt").agg(
    exact_match_accuracy=("exact_match", "mean"),
    average_semantic_similarity=("semantic_similarity", "mean"),
    llm_judge_accuracy=("llm_judge", lambda x: (x.str.upper() == "PASS").mean())
)

summary["exact_match_accuracy"] *= 100
summary["llm_judge_accuracy"] *= 100

print(summary)
print("\nQUESTION-LEVEL COMPARISON")

comparison = df.pivot(
    index="id",
    columns="prompt",
    values="llm_judge"
)

print(comparison)
semantic_comparison = df.pivot(
    index="id",
    columns="prompt",
    values="semantic_similarity"
)

print("\nSEMANTIC SIMILARITY BY QUESTION")
print(semantic_comparison)