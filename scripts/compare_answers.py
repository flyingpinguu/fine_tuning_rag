"""Run with: .venv/bin/python scripts/compare_answers.py

Save unscored baseline and LoRA answers to the same held-out questions.
Run training.py first. No expected answers or evaluation labels enter prompts.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from statistics import median

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from prompts import build_messages
from retrieval import build_index, retrieve, top_k, min_score, model_name as embedding_model, revision as embedding_revision


# 1. Load the trained adapter's settings and check it matches our training data.
project_folder = Path(__file__).resolve().parent.parent
load_dotenv(project_folder / ".env")
adapter_folder = project_folder / os.getenv("ADAPTER_PATH", "artifacts/support-lora-adapter")
with open(adapter_folder / "training_summary.json") as file:
    training_summary = json.load(file)

training_bytes = (project_folder / "data/training/instruction_examples.jsonl").read_bytes()
if hashlib.sha256(training_bytes).hexdigest() != training_summary["training_sha256"]:
    raise ValueError("Training data has changed. Run training.py again.")

max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "80"))
seed = int(os.getenv("RANDOM_SEED", "42"))
set_seed(seed)
device = "mps" if torch.backends.mps.is_available() else "cpu"
timing_repeats = int(os.getenv("TIMING_REPEATS", "1"))
if timing_repeats < 1:
    raise ValueError("TIMING_REPEATS must be positive.")
with open(project_folder / "data/evaluation/cases.json") as file:
    cases = json.load(file)


# 2. Build retrieval once and load the base model plus the saved adapter.
started = perf_counter()
embedding_model_instance, client = build_index()
index_build_seconds = perf_counter() - started
started = perf_counter()
tokenizer = AutoTokenizer.from_pretrained(
    training_summary["base_model"], revision=training_summary["base_model_revision"],
)
base_model = AutoModelForCausalLM.from_pretrained(
    training_summary["base_model"], revision=training_summary["base_model_revision"],
    dtype=torch.float32,
)
model = PeftModel.from_pretrained(base_model, adapter_folder)
model.to(device)
model.eval()  # Turns off dropout for BOTH answer paths.
if device == "mps":
    torch.mps.synchronize()
model_load_seconds = perf_counter() - started

# Warm up each answer path before measuring generation latency.
warmup_inputs = tokenizer("Hello", return_tensors="pt").to(device)
with torch.inference_mode():
    with model.disable_adapter():
        model.generate(**warmup_inputs, max_new_tokens=1, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    model.generate(**warmup_inputs, max_new_tokens=1, do_sample=False, pad_token_id=tokenizer.eos_token_id)
if device == "mps":
    torch.mps.synchronize()


# 3. Retrieve once per question, then give exactly those inputs to both paths.
records = []
for case_number, case in enumerate(cases):
    if device == "mps":
        torch.mps.synchronize()
    started = perf_counter()
    all_hits = retrieve(case["question"], embedding_model_instance, client, apply_threshold=False)
    if device == "mps":
        torch.mps.synchronize()
    retrieval_seconds = perf_counter() - started
    hits = [hit for hit in all_hits if hit.score >= min_score]
    evidence = []
    context_lines = []
    for hit in hits:
        evidence.append({**hit.payload, "score": float(hit.score)})
        context_lines.append(f"[{hit.payload['source_id']}] {hit.payload['text']}")

    record = {"id": case["id"], "question": case["question"], "evidence": evidence}
    record["retrieval_seconds"] = retrieval_seconds
    record["all_retrieved"] = [{**hit.payload, "score": float(hit.score)} for hit in all_hits]
    record["prompt_preparation_seconds"] = 0.0
    if not hits:
        # This is a shared pipeline decision, not a generated model answer.
        record["status"] = "no-evidence"
        record["prompt"] = None
        record["prompt_tokens"] = 0
        for system in ("baseline", "lora"):
            record[system] = {
                "answer": "The approved material is insufficient. Escalate for manual review.",
                "raw_model_output": None,
                "generated_tokens": 0,
                "generation_seconds": 0.0,
                "timing_samples_seconds": [],
                "answer_latency_seconds": retrieval_seconds,
            }
    else:
        record["status"] = "generated"
        context = "\n".join(context_lines)
        started = perf_counter()
        messages = build_messages(case["question"], context)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        if device == "mps":
            torch.mps.synchronize()
        record["prompt_preparation_seconds"] = perf_counter() - started
        prompt_length = inputs["input_ids"].shape[1]
        if prompt_length + max_new_tokens > model.config.max_position_embeddings:
            raise ValueError(f"{case['id']} exceeds the model's context window.")
        record["prompt"] = prompt
        record["prompt_tokens"] = prompt_length

        # Alternate which system runs first to reduce ordering bias in timings.
        systems = ("baseline", "lora") if case_number % 2 == 0 else ("lora", "baseline")
        for system in systems:
            # Greedy decoding uses no sampling. Both calls share the same inputs.
            timing_samples = []
            first_output = None
            for repeat in range(timing_repeats):
                if device == "mps":
                    torch.mps.synchronize()
                started = perf_counter()
                with torch.inference_mode():
                    if system == "baseline":
                        # Temporarily bypass the adapter, keeping original base weights.
                        with model.disable_adapter():
                            output = model.generate(
                                **inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                    else:
                        output = model.generate(
                            **inputs, do_sample=False, max_new_tokens=max_new_tokens,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                if device == "mps":
                    torch.mps.synchronize()
                timing_samples.append(perf_counter() - started)
                if first_output is None:
                    first_output = output.clone()
                elif not torch.equal(first_output, output):
                    raise ValueError(f"Repeated greedy output changed for {case['id']}/{system}.")
            answer_tokens = output[0, prompt_length:]
            answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
            record[system] = {
                "answer": answer,
                "raw_model_output": answer,
                "generated_tokens": len(answer_tokens),
                "generation_seconds": median(timing_samples),
                "timing_samples_seconds": timing_samples,
                "answer_latency_seconds": retrieval_seconds + record["prompt_preparation_seconds"] + median(timing_samples),
            }

    records.append(record)
    print(f"Saved answers for {case['id']}", flush=True)

client.close()


# 4. Write the exact outputs, shared evidence, and configuration. No scoring.
run_time = datetime.now(timezone.utc)
output_folder = project_folder / "artifacts" / run_time.strftime("comparison-%Y%m%d-%H%M%S-%f")
output_folder.mkdir(parents=True)
configuration = {
    "created_at_utc": run_time.isoformat(),
    "training": training_summary,
    "embedding_model": embedding_model,
    "embedding_revision": embedding_revision,
    "evaluation_sha256": hashlib.sha256((project_folder / "data/evaluation/cases.json").read_bytes()).hexdigest(),
    "knowledge_base_sha256": {},
    "top_k": top_k,
    "min_score": min_score,
    "max_new_tokens": max_new_tokens,
    "do_sample": False,
    "device": device,
    "seed": seed,
    "timing_repeats": timing_repeats,
    "index_build_seconds": index_build_seconds,
    "model_load_seconds": model_load_seconds,
    "timing_protocol": "Warm-up each path; alternate system order by case; synchronize MPS around timers; median generation time across repeats. Answer latency = shared retrieval + prompt preparation + median generation; startup excluded.",
}
for file in sorted((project_folder / "data/knowledge_base").glob("*.md")):
    configuration["knowledge_base_sha256"][file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
with open(output_folder / "results.json", "w") as file:
    json.dump({"configuration": configuration, "cases": records}, file, indent=2)

for system, title in (("baseline", "Baseline RAG"), ("lora", "RAG + LoRA"), ("both", "Both systems")):
    with open(output_folder / f"{system}.md", "w") as file:
        file.write(f"# {title}: raw answers\n\n")
        file.write("No scoring or quality assessment has been performed.\n\n")
        file.write(f"Model: `{training_summary['base_model']}`. Adapter trained for {training_summary['epochs']} epochs.\n\n")
        file.write(f"Greedy decoding; maximum {max_new_tokens} new tokens. Identical evidence and prompts for both systems.\n\n")
        file.write("Answers are preserved as generated, without citation filtering or replacement. "
                   "When no evidence meets the threshold, both paths return a pipeline fallback without calling the model.\n\n")
        for record in records:
            file.write(f"## {record['id']}\n\n{record['question']}\n\n")
            file.write("### Shared retrieved evidence\n\n")
            for chunk in record["evidence"]:
                file.write(f"[{chunk['source_id']}] paragraph {chunk['position']}, similarity {chunk['score']:.3f}\n\n")
                file.write(chunk["text"] + "\n\n")
            if record["status"] == "no-evidence":
                file.write("No passage met the threshold. The outputs below are pipeline fallbacks.\n\n")
            systems = ("baseline", "lora") if system == "both" else (system,)
            for name in systems:
                label = "Baseline RAG" if name == "baseline" else "RAG + LoRA"
                file.write(f"### {label}\n\n")
                # Use a fence longer than any backticks in the generated answer.
                answer = record[name]["answer"]
                fence = "`" * max(3, len(answer) - len(answer.replace("`", "")) + 1)
                file.write(f"{fence}text\n{answer}\n{fence}\n\n")

print(f"Results written to {output_folder}")
