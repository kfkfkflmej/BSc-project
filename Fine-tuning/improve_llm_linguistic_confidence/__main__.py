import os
import torch
import logging

from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


logging.basicConfig(level=logging.INFO)

# =========================
# Config
# =========================
model_id = "google/gemma-4-E2B-it"  # or E4B if you have more VRAM

system_message = "You are an expert product description writer for Amazon."

user_prompt = """Create a Short Product description based on the provided <PRODUCT> and <CATEGORY>.
Only return description. The description should be SEO optimized and for a better mobile search experience.

<PRODUCT>
{product}
</PRODUCT>

<CATEGORY>
{category}
</CATEGORY>
"""


# =========================
# Dataset formatting (TEXT ONLY)
# =========================
def format_data(sample):
    return {
        "messages": [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": user_prompt.format(
                    product=sample["Product Name"],
                    category=sample["Category"],
                ),
            },
            {
                "role": "assistant",
                "content": sample["description"].strip(),
            },
        ],
    }


# =========================
# Load dataset
# =========================
dataset = load_dataset(
    "philschmid/amazon-product-descriptions-vlm",
    split="train"
)

dataset = dataset.train_test_split(test_size=0.1)

dataset_train = [format_data(x) for x in dataset["train"]]
dataset_test = [format_data(x) for x in dataset["test"]]

logging.info(f"Train size: {len(dataset_train)}")
logging.info(f"Test size: {len(dataset_test)}")


# =========================
# Processor
# =========================
processor = AutoProcessor.from_pretrained(model_id)

# ensure padding token exists
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token


# =========================
# Quantization (QLoRA)
# =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


# =========================
# Model
# =========================
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)


# =========================
# LoRA config
# =========================
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"],
)


# =========================
# Collate function (TEXT ONLY)
# =========================
def collate_fn(examples):
    texts = []

    for example in examples:
        text = processor.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text.strip())

    batch = processor(
        text=texts,
        padding=True,
        return_tensors="pt",
    )

    labels = batch["input_ids"].clone()

    # mask padding tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100

    batch["labels"] = labels
    return batch


# =========================
# Training config
# =========================
training_args = SFTConfig(
    output_dir="gemma-text-product-desc",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    bf16=True,
    max_grad_norm=0.3,
    warmup_steps=50,
    lr_scheduler_type="constant",
    report_to="tensorboard",

    # important for custom collator
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
)


# =========================
# Trainer
# =========================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
)


# =========================
# Debug sample
# =========================
sample_text = processor.apply_chat_template(
    dataset_train[0]["messages"],
    tokenize=False
)

print("\n===== SAMPLE TRAIN TEXT =====\n")
print(sample_text)
print("\n============================\n")


# =========================
# Train
# =========================
trainer.train()


# =========================
# Save model
# =========================
trainer.save_model("gemma-text-final")

logging.info("Training complete and model saved.")