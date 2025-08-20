import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --------------------
# Model + Tokenizer
# --------------------
model_name = "mistralai/Mistral-7B-Instruct-v0.1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare model for k-bit training + add LoRA
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# --------------------
# Ensure gradients are enabled
# --------------------
model.train()
for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True

model.print_trainable_parameters()  # Debug

# --------------------
# Dataset
# --------------------
dataset = load_dataset("json", data_files="/workspace/single_person_dataset.jsonl", split="train")

def tokenize(batch):
    texts = [p + " " + c for p, c in zip(batch["prompt"], batch["completion"])]
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

dataset = dataset.map(tokenize, batched=True, remove_columns=["prompt", "completion"])

# --------------------
# Training Args
# --------------------
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    optim="paged_adamw_32bit",
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    report_to="none",
    gradient_checkpointing=True
)

# --------------------
# Trainer
# --------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer  # replaces deprecated tokenizer arg
)

model.config.use_cache = False  # Needed for checkpointing

# --------------------
# Train
# --------------------
trainer.train()
