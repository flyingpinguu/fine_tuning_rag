"""Run with: .venv/bin/python scripts/training.py

Train a LoRA adapter on the supplied examples. The first run downloads SmolLM2.
Evaluation cases are not used here. Defaults can be changed in .env.
"""

import hashlib
import json
import os
import runpy
from pathlib import Path
from time import perf_counter

import torch
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from prompts import build_messages


# 1. Settings. These match the reference solution's starting configuration.
project_folder = Path(__file__).resolve().parent.parent
load_dotenv(project_folder / ".env")

model_name = os.getenv("BASE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
revision = os.getenv("BASE_MODEL_REVISION") or None
if model_name == "HuggingFaceTB/SmolLM2-135M-Instruct" and revision is None:
    revision = "12fd25f77366fa6b3b4b768ec3050bf629380bac"

epochs = int(os.getenv("TRAIN_EPOCHS", "8"))
learning_rate = float(os.getenv("LEARNING_RATE", "0.0002"))
max_length = int(os.getenv("MAX_SEQUENCE_LENGTH", "256"))
seed = int(os.getenv("RANDOM_SEED", "42"))
adapter_folder = project_folder / os.getenv("ADAPTER_PATH", "artifacts/support-lora-adapter")

lora_rank = 8
lora_alpha = 16
lora_dropout = 0.05
target_modules = ["q_proj", "v_proj"]

if epochs < 1 or learning_rate <= 0 or max_length < 1:
    raise ValueError("Epochs, learning rate, and maximum length must be positive.")
set_seed(seed)
device = "mps" if torch.backends.mps.is_available() else "cpu"


# 2. Run our validation script, then read only the training examples.
runpy.run_path(str(project_folder / "scripts/validate_data.py"))
training_file = project_folder / "data/training/instruction_examples.jsonl"
training_bytes = training_file.read_bytes()
examples = []
for line in training_bytes.decode("utf-8").splitlines():
    if line.strip():
        examples.append(json.loads(line))


# 3. Turn each context + question + answer into model tokens.
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)


training_batches = []
for example in examples:
    messages = build_messages(example["question"], example["context"])
    # The prompt ends where the model should start writing its answer.
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    messages.append({"role": "assistant", "content": example["answer"]})
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)

    batch = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    prompt_length = prompt_ids.shape[1]

    if batch["input_ids"].shape[1] > max_length:
        raise ValueError(f"{example['id']} exceeds {max_length} tokens. Increase MAX_SEQUENCE_LENGTH.")
    if not torch.equal(batch["input_ids"][:, :prompt_length], prompt_ids):
        raise ValueError("The training text must begin with the inference prompt.")

    # Labels are the token targets. -100 tells the loss to ignore prompt tokens.
    # We train the model to produce the answer, rather than reproduce the question.
    labels = batch["input_ids"].clone()
    labels[:, :prompt_length] = -100
    if (labels != -100).sum().item() == 0:
        raise ValueError(f"{example['id']} has no answer tokens to learn from.")
    batch["labels"] = labels
    training_batches.append(batch)


# 4. Load the original model and attach small trainable LoRA matrices.
print(f"Loading {model_name} on {device}...", flush=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name, revision=revision, dtype=torch.float32,
)
base_model.loss_type = "ForCausalLM"
base_model.config.use_cache = False  # Generation caching is unnecessary during training.
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    target_modules=target_modules,
    r=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
)
model = get_peft_model(base_model, lora_config)
model.to(device)
model.print_trainable_parameters()

# PEFT freezes the base weights. The optimizer receives only trainable weights.
trainable_parameters = []
for parameter in model.parameters():
    if parameter.requires_grad:
        trainable_parameters.append(parameter)
optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=0.01)


# 5. Training loop: one example per batch, one update per example.
model.train()  # Enables training behavior, including dropout.
epoch_losses = []
started = perf_counter()
for epoch in range(epochs):
    total_loss = 0.0
    for batch in training_batches:
        optimizer.zero_grad()  # Clear gradients from the previous example.
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        loss = outputs.loss
        loss.backward()  # Calculate gradients for the LoRA weights.
        optimizer.step()  # Use those gradients to update the LoRA weights.
        total_loss += loss.item()

    average_loss = total_loss / len(training_batches)
    epoch_losses.append(average_loss)
    print(f"Epoch {epoch + 1}/{epochs}: average training loss = {average_loss:.4f}", flush=True)


# 6. Save the adapter (not another full copy of the base model) and run settings.
adapter_folder.mkdir(parents=True, exist_ok=True)
model.save_pretrained(adapter_folder, save_embedding_layers=False)
tokenizer.save_pretrained(adapter_folder)
trainable_count, total_count = model.get_nb_trainable_parameters()
summary = {
    "base_model": model_name,
    "base_model_revision": revision,
    "training_sha256": hashlib.sha256(training_bytes).hexdigest(),
    "adapter_path": str(adapter_folder),
    "epochs": epochs,
    "learning_rate": learning_rate,
    "max_sequence_length": max_length,
    "random_seed": seed,
    "batch_size": 1,
    "weight_decay": 0.01,
    "lora_target_modules": target_modules,
    "lora_rank": lora_rank,
    "lora_alpha": lora_alpha,
    "lora_dropout": lora_dropout,
    "trainable_parameters": trainable_count,
    "total_parameters": total_count,
    "trainable_percentage": 100 * trainable_count / total_count,
    "epoch_losses": epoch_losses,
    "training_seconds": perf_counter() - started,
    "training_device": device,
    "prompt_format": "Native tokenizer chat template; system + context/question + answer",
}
with open(adapter_folder / "training_summary.json", "w") as file:
    json.dump(summary, file, indent=2)

print(f"Saved adapter and training summary to {adapter_folder}")
print("Training loss measures fit to training examples. Evaluate held-out answers separately.")
