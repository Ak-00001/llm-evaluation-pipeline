# LLM Evaluation Pipeline

An automated evaluation pipeline for comparing LLM outputs across different prompts using multiple evaluation methods.

## Overview

This project evaluates LLM responses against a golden dataset using three complementary evaluation methods:

* **Exact Match** - checks whether the normalized answer matches the expected answer.
* **Semantic Similarity** - compares the meaning of expected and actual answers using sentence embeddings.
* **LLM-as-a-Judge** - uses an LLM to determine whether the generated answer should be accepted.

The pipeline also uses **Pandas** for experiment analysis and **Langfuse** for LLM observability.

## Architecture

```text
Golden Dataset
      │
      ▼
   LLM Runner
      │
      ├──────────────► Langfuse Observability
      │
      ▼
Generated Answers
      │
      ├──► Exact Match
      │
      ├──► Semantic Similarity
      │
      └──► LLM-as-a-Judge
      │
      ▼
   results.csv
      │
      ▼
Pandas Analysis
      │
      ▼
Prompt Comparison
```

## Features

* Automated LLM evaluation
* Prompt A/B comparison
* Exact-match evaluation
* Embedding-based semantic similarity
* LLM-as-a-judge evaluation
* Experiment analysis with Pandas
* Langfuse LLM observability
* Request logging
* API error handling
* Automated tests with pytest
* CSV-based golden dataset and results

## Tech Stack

* Python
* Ollama Cloud API
* `gpt-oss:20b`
* Sentence Transformers
* Pandas
* Langfuse
* Requests
* Pytest

## Project Structure

```text
llm-evaluation-pipeline/
│
├── data/
│   ├── golden_dataset.csv
│   └── results.csv
│
├── src/
│   ├── main.py
│   ├── evaluation.py
│   └── analyze_results.py
│
├── tests/
│   └── test_evaluation.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Evaluation Methods

### 1. Exact Match

The expected and generated answers are normalized before comparison.

Normalization removes:

* Leading/trailing whitespace
* Case differences
* Common punctuation
* Duplicate whitespace

For example:

```text
Expected: Berlin
Actual:   Berlin.
```

is considered a match.

### 2. Semantic Similarity

Sentence Transformers are used to generate embeddings for the expected and generated answers.

Cosine similarity is then used to measure how semantically similar the two answers are.

This allows different wording to receive a high similarity score.

### 3. LLM-as-a-Judge

An evaluation prompt asks an LLM to classify the generated response as:

```text
PASS
```

or

```text
FAIL
```

The judge considers whether the answer correctly addresses the question and is consistent with the expected answer.

## Prompt Experiment

The pipeline evaluates two different prompts:

* **Prompt A:** concise question answering
* **Prompt B:** more explicit instructions for direct and precise answers

The same golden dataset is evaluated with both prompts.

The resulting metrics are aggregated with Pandas to compare their observed performance.

Because LLM outputs can vary between runs, the results represent an experiment on the current dataset rather than a universal ranking of the prompts.

## Example Analysis

A sample experiment produced results similar to:

| Prompt   | Exact Match | LLM Judge |
| -------- | ----------: | --------: |
| Prompt A |       33.3% |      100% |
| Prompt B |        0.0% |      100% |

Semantic similarity is also calculated for every question and aggregated by prompt.

The small dataset is intentionally simple for demonstrating the evaluation pipeline. A production evaluation system would use a substantially larger and more diverse golden dataset.

## Observability

Langfuse is integrated into the LLM generation process to record information such as:

* Model
* Input
* Output
* Generation traces
* Request latency
* Token/cost information when provided by the provider

This separates two concerns:

**Evaluation:**
"Was the generated answer good?"

**Observability:**
"What happened when the LLM ran?"

## Testing

The project includes automated tests using Pytest.

Run:

```bash
python -m pytest
```

Current tests cover:

* Answer normalization
* Correct exact matches
* Incorrect exact matches
* Semantic similarity

Example:

```text
4 passed
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Ak-00001/llm-evaluation-pipeline.git
cd llm-evaluation-pipeline
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file:

```text
OLLAMA_API_KEY=your_ollama_api_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

Never commit `.env` or expose API keys publicly.

### 5. Run the evaluation

```bash
python src/main.py
```

The generated evaluation results are saved to:

```text
data/results.csv
```

### 6. Analyze the results

```bash
python src/analyze_results.py
```

## Future Improvements

Potential extensions include:

* Larger and more diverse evaluation datasets
* Separate judge models
* Additional evaluation metrics
* Statistical analysis across repeated runs
* Experiment tracking across model versions
* CI/CD integration
* Containerized deployment

## Author

**Ayush Kshatri**

M.Sc. Applied Computer Science
Hochschule Schmalkalden
