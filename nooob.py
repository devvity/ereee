# train_nemotron_lora.py
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType

# -------------r----------------
# 0. CUDA memory fragmentation fix
# -----------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -----------------------------
# 1. Load dataset (JSONL format)
# -----------------------------
dataset = load_dataset(
    "json",
    data_files="/workspace/single_person_dataset.jsonl",
    split="train"
)

# -----------------------------
# 2. Load tokenizer and model (QLoRA)
# -----------------------------
model_name = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config
)

# enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# -----------------------------
# 3. Tokenize dataset
# -----------------------------
def tokenize(batch):
    # Each batch["prompt"] is a list of strings, and batch["completion"] is a list of strings
    texts = [p + " " + c for p, c in zip(batch["prompt"], batch["completion"])]
    
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    
    # Add labels = input_ids for causal LM training
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized



dataset = dataset.map(tokenize, batched=True, remove_columns=["prompt", "completion"])


# -----------------------------
# 4. LoRA configuration
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # recommended for Mistral
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# -----------------------------
# 5. Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./lora_mistral",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=200,
    save_total_limit=2,
    fp16=False,
    bf16=True,
    optim="paged_adamw_32bit",
    report_to="none"
)

# -----------------------------
# 6. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# -----------------------------
# 7. Start training
# -----------------------------
trainer.train()
