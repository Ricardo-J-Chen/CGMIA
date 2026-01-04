import json
from contextlib import contextmanager
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


data = json.load(open('codemi.json', 'r'))
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_paths = ["../models/shadow_model_1_finetuned", "./model/shadow_model_2_finetuned", "./model/target_model_finetuned"]
# model_paths = "./model/target_model_finetuned"

# Define keywords to skip
keywords = {'', '(', ')', '{', '}', '_', "auto", "break", "case", "char", "const", "continue", "default", "do",
            "double", "else", "enum", "extern", "float", "for", "goto", "if",
            "inline", "int", "long", "register", "restrict", "return", "short",
            "signed", "sizeof", "static", "struct", "switch", "typedef", "union",
            "unsigned", "void", "volatile", "while", "_Alignas", "_Alignof",
            "_Atomic", "_Bool", "_Complex", "_Generic", "_Imaginary", "_Noreturn",
            "_Static_assert", "_Thread_local"}

# Global cache for models and tokenizers
MODEL_CACHE = {}

@contextmanager
def clear_cuda_cache():
    """Context manager for clearing CUDA cache"""
    try:
        yield
    finally:
        torch.cuda.empty_cache()

def load_model(model_name, device='cuda'):
    """Load and cache model"""
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    tokenizer_path = f"../models/{model_name}"
    model_path = f"../models/{model_name}"

    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load LoRA adapter
    checkpoints = {
        "codegemma-2b": 6000,
        "deepseek-coder-1.3b-instruct": 7000,
        "Qwen2.5-Coder-3B-Instruct": 4000
    }
    model_id = f"../models/{model_name}-lora/checkpoint-{checkpoints[model_name]}"
    p_model = PeftModel.from_pretrained(model, model_id)

    # Cache model
    MODEL_CACHE[model_name] = (tokenizer, p_model)

    return tokenizer, p_model

def calculate_ranks(model_name, text, device='cuda:1'):
    """
    Calculate perplexity for given text (fixed memory leak version)

    Args:
        model_name: model name
        text: text to calculate perplexity for
        device: computation device

    Returns:
        perplexity value
    """
    # Use context manager to ensure resource cleanup
    with clear_cuda_cache(), torch.no_grad():
        # Get or load model
        tokenizer, p_model = load_model(model_name, device)
        # Encode text and prepare input
        encodings = tokenizer(text, return_tensors='pt')
        input_ids = encodings.input_ids.to(device)

        ranks = []
        # Get token IDs directly instead of converting one by one
        tokens = input_ids[0].tolist()
        cleaned_tokens = [tokenizer.convert_ids_to_tokens(token).replace('Ġ', '').replace('Ċ', '') for token in tokens]

        seq_len = input_ids.shape[1]  # Get sequence length

        past_key_values = None  # For caching past_key_values (only for Transformer-based models)

        for i in range(1, seq_len):
            # Only input sequence up to current token
            current_input_ids = input_ids[:, :i]

            with torch.no_grad():
                # Only forward sequence up to current token
                if past_key_values is None:
                    outputs = p_model(current_input_ids)
                else:
                    outputs = p_model(input_ids[:, i - 1].unsqueeze(0), past_key_values=past_key_values)

                logits = outputs.logits  # Get prediction logits
                past_key_values = outputs.past_key_values  # Only cache past key-values

            token = cleaned_tokens[i]  # Current token
            real_token_id = input_ids[0, i]  # Real token ID

            # Skip rank calculation for keywords
            if token in keywords:
                continue

            probs = logits[0, -1, :]  # Get last predicted probability distribution
            rank = torch.sum(probs > probs[real_token_id]).item()  # Calculate real token rank
            # print(f"Rank for {token}: {rank}")

            ranks.append(rank)

        # Explicitly release no longer needed variables
        del input_ids, outputs, encodings

        return ranks

from tqdm import tqdm
for i in tqdm(range(len(data))):
    if "ranks" not in data[i] and data[i]["index"]==1:
        data[i]["ranks"] = calculate_ranks(data[i]["model"],data[i]["code"])
    else:
        continue
with open('codemi.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=4, ensure_ascii=False)