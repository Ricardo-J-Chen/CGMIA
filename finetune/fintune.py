from transformers import AutoTokenizer,AutoModelForCausalLM
import os
from tqdm import tqdm
import json
import torch
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset

from peft import LoraConfig,get_peft_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device:{device}")

# model_name = "codegemma-2b"
# model_name = "Qwen2.5-Coder-3B-Instruct"
model_path = f"./models/{model_name}"
tokenizer_path = f"./models/{model_name}"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,torch_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,device_map="auto")

tokenizer.pad_token = tokenizer.eos_token
# Get GPU memory usage by current model (difference is memory reserved for PyTorch)
memory_footprint_bytes = model.get_memory_footprint()
memory_footprint_mib = memory_footprint_bytes/(1024**3)
print(f"Model memory footprint: {memory_footprint_mib:.2f} GB")

# Create a LoraConfig object to set LoRA configuration parameters
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj","c_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.04,
    bias='none',
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
# Print trainable parameters in the model
model.print_trainable_parameters()

with open("./train_data.json","r",encoding="utf-8") as f:
    train_data = json.load(f)

# Ensure all fields are strings
for item in train_data:
    item["task_id"] = str(item["task_id"])  # Force convert task_id to string

dataset=Dataset.from_list(train_data)


def process_func_batched(examples):
    batch = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }

    MAX_LENGTH = 2048

    for i in range(len(examples["task_id"])):
        instruction_text = f"{examples['prompt'][i]}\n"
        response_text = f"{examples['solution'][i]}"

        # Use padding and truncation
        instruction = tokenizer(
            instruction_text,
            add_special_tokens=False,
            max_length = MAX_LENGTH,
            truncation=True
        )
        response = tokenizer(
            response_text,
            add_special_tokens=False,
            max_length=MAX_LENGTH,
            truncation=True
        )

        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

        # Ensure not exceeding max length
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]

        batch["input_ids"].append(input_ids)
        batch["attention_mask"].append(attention_mask)
        batch["labels"].append(labels)

    # Pad the batch
    batch["input_ids"] = [seq + [tokenizer.pad_token_id] * (MAX_LENGTH - len(seq)) for seq in batch["input_ids"]]
    batch["attention_mask"] = [seq + [0] * (MAX_LENGTH - len(seq)) for seq in batch["attention_mask"]]
    batch["labels"] = [seq + [-100] * (MAX_LENGTH - len(seq)) for seq in batch["labels"]]

    return batch


# Use batched processing
tokenized_dataset = dataset.map(
    process_func_batched,
    batched=True,
    batch_size=8  # Can be adjusted based on memory
)

model_dir = "models"

data_collator = DataCollatorForLanguageModeling(tokenizer,mlm=False)

training_args = TrainingArguments(
    output_dir=f"{model_dir}/{model_name}-lora",
    per_device_train_batch_size=1,
    learning_rate=5e-5,
    fp16=True,
    logging_steps=20,
    num_train_epochs=20,
)
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    data_collator=data_collator
)

trainer.train()