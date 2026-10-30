"""Run with: .venv/bin/python scripts/validate_data.py"""

import json
import re
from pathlib import Path


# Find the data folder relative to this script, even if we run it elsewhere.
project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data"

# JSONL stores one training example per line.
training_examples = []
with open(data_folder / "training/instruction_examples.jsonl") as file:
    for line in file:
        if line.strip():  # Skip empty lines.
            training_examples.append(json.loads(line))

# The evaluation JSON file stores all cases in one list.
with open(data_folder / "evaluation/cases.json") as file:
    evaluation_cases = json.load(file)

# A filename such as "refunds.md" gives us the source ID "refunds".
source_ids = []
for file in (data_folder / "knowledge_base").glob("*.md"):
    source_ids.append(file.stem)

# Stop if an input is empty, so an empty dataset cannot pass the checks.
if not training_examples or not evaluation_cases or not source_ids:
    raise ValueError("Training data, evaluation data, and knowledge base must not be empty.")


# 1. Every training example must contain these fields with a value.
required_fields = ["id", "context", "question", "answer", "reference_ids"]

for example in training_examples:
    for field in required_fields:
        if field not in example or not example[field]:
            raise ValueError(f"Training example is missing {field}: {example}")

print("Check 1 passed: training examples have all required fields.")


# 2. Every declared or cited source must exist in the knowledge base.
for example in training_examples:
    for source_id in example["reference_ids"]:
        if source_id not in source_ids:
            raise ValueError(f"Unknown training source: {source_id}")

    # Find text inside square brackets: "See [refunds]" becomes ["refunds"].
    text = example["context"] + "\n" + example["answer"]
    citations = re.findall(r"\[([^\[\]\n]+)\]", text)
    for source_id in citations:
        if source_id not in source_ids:
            raise ValueError(f"Unknown citation: {source_id}")

for case in evaluation_cases:
    for source_id in case["expected_sources"]:
        if source_id not in source_ids:
            raise ValueError(f"Unknown evaluation source: {source_id}")

print("Check 2 passed: all source IDs exist in the knowledge base.")


# 3. Evaluation IDs must be unique.
evaluation_ids = []
for case in evaluation_cases:
    if case["id"] in evaluation_ids:
        raise ValueError(f"Duplicate evaluation ID: {case['id']}")
    evaluation_ids.append(case["id"])

print("Check 3 passed: evaluation IDs are unique.")


# 4. Evaluation cases must not appear in training.
# Compare IDs and question text. This does not detect paraphrased questions.
for case in evaluation_cases:
    for example in training_examples:
        if case["id"] == example["id"]:
            raise ValueError(f"ID appears in both datasets: {case['id']}")

        # Ignore capitalization and extra whitespace when comparing questions.
        evaluation_question = " ".join(case["question"].lower().split())
        training_question = " ".join(example["question"].lower().split())
        if evaluation_question == training_question:
            raise ValueError(f"Evaluation question appears in training: {case['id']}")

print("Check 4 passed: no shared IDs or matching questions between datasets.")
print("All data validation checks passed!")
